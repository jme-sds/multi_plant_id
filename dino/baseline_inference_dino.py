import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import glob
import csv
from collections import defaultdict
import math

# ==========================================
# CONFIGURATION
# ==========================================
# Paths
PLANT_HOME = "/scratch/jme3qd/data/plantclef2025" # Base directory for models

TEST_IMAGES_PATH = os.path.join(PLANT_HOME,"quadrat/images") #"./data/plantclef2025_test_quadrats" 
CLASS_LIST_DIR = os.path.join(PLANT_HOME,"images_max_side_800") # "./data/plantclef_single_images" # Needed to map index -> species ID
#PLANT_HOME = "./data" # Directory containing dinov2_model/model_best.pth.tar

# Model Config
MODEL_CHECKPOINT = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
IMAGE_SIZE = 518
GRID_SIZE = (4, 4) # Rows, Cols

# Hardware Config
# A6000 has 48GB VRAM. We can push batch size.
# DINOv2-Base (518px) takes roughly 1.5GB-2GB per sample in training, less in inference.
# Safe start: 32 per GPU * 4 GPUs = 128.
BATCH_SIZE = 512 
NUM_WORKERS = 16 # High worker count for fast IO on 4 GPUs

# Checkpoint to load
# Option A: The raw backbone from organizers (Classifier head will be random!)
BACKBONE_PATH = os.path.join(PLANT_HOME, "dinov2_model/model_best.pth.tar")
#BACKBONE_PATH = "./baseline_fine_tuned.pth"
# Option B: Your fine-tuned model (uncomment and set path if you have one)
TRAINED_MODEL_PATH = "./baseline_fine_tuned.pth" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()

print(f"Running on {DEVICE} with {NUM_GPUS} GPUs.")

# ==========================================
# DATASET
# ==========================================

class QuadratTilingDataset_Inference(Dataset):
    """
    Tiles high-res quadrat images into a fixed grid for inference.
    """
    def __init__(self, data_dir, grid_size=(3, 3), transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        self.num_rows, self.num_cols = grid_size
        self.num_tiles = self.num_rows * self.num_cols
        
        self._create_samples()

    def _create_samples(self):
        img_paths = []
        if not os.path.exists(self.data_dir):
            print(f"Warning: Test data directory {self.data_dir} does not exist.")
            return

        # Find all common image types
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.JPG", "*.JPEG"]
        for ext in extensions:
             img_paths.extend(glob.glob(os.path.join(self.data_dir, ext)))
        
        # Sort for reproducibility
        img_paths = sorted(img_paths)

        for img_path in img_paths:
            for i in range(self.num_tiles):
                self.samples.append((img_path, i))
                
        print(f"Initialized Dataset: {len(img_paths)} images -> {len(self.samples)} tiles.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tile_index = self.samples[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            tile = self._get_tile(img, tile_index)

            if self.transform:
                tile = self.transform(tile)
            
            return tile, img_path

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), "error"

    def _get_tile(self, img, tile_index):
        img_width, img_height = img.size
        
        tile_width = img_width // self.num_cols
        tile_height = img_height // self.num_rows
        
        row = tile_index // self.num_cols
        col = tile_index % self.num_cols
        
        left = col * tile_width
        top = row * tile_height
        
        # Handle edge pixels
        right = img_width if col == self.num_cols - 1 else (col + 1) * tile_width
        bottom = img_height if row == self.num_rows - 1 else (row + 1) * tile_height
            
        return img.crop((left, top, right, bottom))

# ==========================================
# MODEL WRAPPER
# ==========================================

class SimpleClassifier(nn.Module):
    """
    Baseline architecture: Frozen Backbone + Linear Head
    """
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        
        # Auto-detect embedding dimension
        if hasattr(backbone, "embed_dim"):
            dim = backbone.embed_dim
        elif hasattr(backbone, "num_features"):
            dim = backbone.num_features
        else:
            # Fallback for DINOv2-base
            dim = 768
            
        self.classifier = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        # timm forward_features returns [Batch, Tokens, Dim]
        features = self.backbone.forward_features(x)
        # Use CLS token (Index 0)
        cls_token = features[:, 0]
        logits = self.classifier(cls_token)
        return logits

# ==========================================
# SETUP UTILS
# ==========================================

def load_class_mapping(root_dir):
    if not os.path.exists(root_dir):
        print(f"Warning: Class directory {root_dir} not found. Creating dummy classes.")
        return [str(i) for i in range(100)]
    
    # Classes are usually sorted folder names
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"Loaded {len(classes)} classes.")
    return classes

def load_model(num_classes):
    print(f"Creating model: {MODEL_CHECKPOINT}")
    base_model = timm.create_model(MODEL_CHECKPOINT, pretrained=True)
    
    # 1. Load Backbone Weights (Challenge Checkpoint)
    if os.path.exists(BACKBONE_PATH):
        print(f"Loading backbone weights from {BACKBONE_PATH}")
        checkpoint = torch.load(BACKBONE_PATH, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Load backbone
        missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
        print(f"Backbone Loaded. Missing keys: {len(missing)}")
    else:
        print("Backbone checkpoint not found. Using timm default pretrained weights.")

    # 2. Wrap in Classifier
    model = SimpleClassifier(base_model, num_classes)
    
    # 3. (Optional) Load Full Fine-Tuned Model
    # If you have a trained .pth file, uncomment the lines below:
    if 'TRAINED_MODEL_PATH' in globals() and os.path.exists(TRAINED_MODEL_PATH):
        print(f"Loading fine-tuned state from {TRAINED_MODEL_PATH}")
        state = torch.load(TRAINED_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
    
    model.to(DEVICE)
    model.eval()
    
    # 4. Wrap for Multi-GPU
    if NUM_GPUS > 1:
        print(f"Wrapping model in DataParallel for {NUM_GPUS} GPUs.")
        model = nn.DataParallel(model)
        
    return model

def write_submission(pred_dict, out_csv):
    print(f"Writing submission to {out_csv}...")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quadrat_id", "species_ids"])
        for quad, species in pred_dict.items():
            s = "[" + ", ".join(str(x) for x in species) + "]"
            writer.writerow([quad, s])
    print("Done.")

# ==========================================
# MAIN
# ==========================================

def main():
    # 1. Prepare Data
    classes = load_class_mapping(CLASS_LIST_DIR)
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = QuadratTilingDataset_Inference(
        TEST_IMAGES_PATH, 
        grid_size=GRID_SIZE, 
        transform=transform
    )
    
    if len(dataset) == 0:
        print("No images found. Exiting.")
        return

    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    # 2. Load Model
    model = load_model(len(classes))

    # 3. Inference Loop
    print("\nStarting Inference...")
    quadrat_preds = defaultdict(set)
    
    with torch.no_grad():
        for tiles, paths in tqdm(loader, desc="Processing"):
            tiles = tiles.to(DEVICE)
            
            # Forward Pass
            logits = model(tiles)
            
            # Get Predictions (Argmax)
            preds = torch.argmax(logits, dim=1)
            preds_cpu = preds.cpu().numpy()
            
            # Map back to Quadrat ID
            for i, path in enumerate(paths):
                if path == "error": continue
                
                pred_idx = preds_cpu[i]
                species_id = classes[pred_idx]
                
                # Filename -> Quadrat ID
                filename = os.path.basename(path)
                quadrat_id = os.path.splitext(filename)[0]
                
                quadrat_preds[quadrat_id].add(species_id)

    # 4. Format and Write
    final_preds = {k: sorted(list(v)) for k, v in quadrat_preds.items()}
    
    out_filename = f"submission_baseline_{GRID_SIZE[0]}x{GRID_SIZE[1]}.csv"
    write_submission(final_preds, out_filename)

if __name__ == "__main__":
    main()
