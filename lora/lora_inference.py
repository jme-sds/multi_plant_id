import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from tqdm import tqdm
import timm
import glob
import csv
from collections import defaultdict
import argparse

# ==========================================
# CONFIGURATION & ARGS
# ==========================================
PLANT_HOME = os.getenv("PLANT_HOME")

def parse_args():
    parser = argparse.ArgumentParser(description="PlantCLEF LoRA Inference Script")
    parser.add_argument("--grid_size", type=int, nargs=2, default=(3, 4), help="Grid size for inference (rows cols), e.g. 3 4")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--model_path", type=str, default="lucas/models/final_lora_classifier_model.pth", help="Relative path to the final .pth model inside PLANT_HOME")
    parser.add_argument("--backbone_path", type=str, default="dinov2_model/model_best.pth.tar", help="Relative path to challenge backbone weights")
    parser.add_argument("--classes_dir", type=str, default="images_max_side_800", help="Folder containing class subfolders (to derive class names)")
    parser.add_argument("--test_dir", type=str, default="quadrat/images", help="Folder containing test quadrat images")
    return parser

# ==========================================
# DATASET
# ==========================================

class QuadratTilingDataset_Inference(Dataset):
    def __init__(self, data_dir, grid_size=(3, 3), transform=None, img_size=518):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.num_rows, self.num_cols = grid_size
        self.num_tiles = self.num_rows * self.num_cols
        self.img_size = img_size
        self._create_samples()

    def _create_samples(self):
        img_paths = []
        if not os.path.exists(self.data_dir):
            print(f"Warning: Test data directory {self.data_dir} does not exist.")
            return
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.JPG", "*.JPEG"]:
             img_paths.extend(glob.glob(os.path.join(self.data_dir, ext)))
        for img_path in sorted(img_paths): 
            for i in range(self.num_tiles): 
                self.samples.append((img_path, i))
        print(f"Found {len(img_paths)} images, creating {len(self.samples)} total tiles.")

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
            print(f"Error loading tile for image {img_path} at index {idx}: {e}")
            dummy_tile = torch.zeros((3, self.img_size, self.img_size)) 
            return dummy_tile, "error_path"

    def _get_tile(self, img, tile_index):
        img_width, img_height = img.size
        tile_width = img_width // self.num_cols
        tile_height = img_height // self.num_rows
        row = tile_index // self.num_cols
        col = tile_index % self.num_cols
        left = col * tile_width
        top = row * tile_height
        if col == self.num_cols - 1: right = img_width
        else: right = (col + 1) * tile_width
        if row == self.num_rows - 1: bottom = img_height
        else: bottom = (row + 1) * tile_height
        tile = img.crop((left, top, right, bottom))
        return tile

# ==========================================
# MODEL WRAPPER
# ==========================================

class LoRAClassifier(nn.Module):
    """
    Wraps backbone + classifier.
    Structure matches the training script to ensure state_dict keys align.
    """
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        # Dynamic dim check
        dim = backbone.base_model.model.embed_dim if hasattr(backbone, 'base_model') else backbone.embed_dim
        self.classifier = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        logits = self.classifier(cls_token)
        return logits

# ==========================================
# UTILS
# ==========================================

def get_class_names(classes_dir):
    if not os.path.exists(classes_dir):
        raise ValueError(f"Classes directory not found: {classes_dir}")
    classes = sorted([d for d in os.listdir(classes_dir) if os.path.isdir(os.path.join(classes_dir, d))])
    print(f"Loaded {len(classes)} classes from {classes_dir}")
    return classes

def load_lora_model(model_checkpoint, backbone_weights_path, trained_model_path, num_classes, device):
    print("Loading base timm model...")
    model = timm.create_model(model_checkpoint, pretrained=True)
    
    # 1. Load Challenge Backbone Weights
    if os.path.exists(backbone_weights_path):
        print(f"Loading backbone weights from {backbone_weights_path}")
        checkpoint = torch.load(backbone_weights_path, map_location=device, weights_only=False) 
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Backbone weights not found at {backbone_weights_path}, using default pretrained.")

    # 2. Inject LoRA Config (Initialize architecture)
    # Note: We don't load adapters here; we initialize them so the state_dict keys exist
    print("Initializing LoRA adapters...")
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["qkv"], 
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, config)
    
    # 3. Wrap in Classifier (Initialize head)
    model = LoRAClassifier(model, num_classes)
    
    # 4. Load Full Fine-Tuned State Dict (Adapters + Head)
    if os.path.exists(trained_model_path):
        print(f"Loading fine-tuned model (adapters + head) from {trained_model_path}")
        full_state = torch.load(trained_model_path, map_location=device)
        keys = model.load_state_dict(full_state, strict=False)
        print(f"Weights loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")
    else:
        raise FileNotFoundError(f"Trained model not found at {trained_model_path}")

    model.to(device)
    model.eval()
    return model

def write_submission(pred_dict, out_csv):
    print(f"Writing submission to {out_csv}")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quadrat_id", "species_ids"])
        for quad, species in pred_dict.items():
            s = "[" + ", ".join(str(x) for x in species) + "]"
            writer.writerow([quad, s])

def run_inference(model, loader, class_names, device, output_file, num_gpus):
    print(f"\n--- Running Inference ---")
    
    # Wrap for Multi-GPU Inference
    if num_gpus > 1 and not isinstance(model, nn.DataParallel):
        print(f"Parallelizing inference across {num_gpus} GPUs.")
        model = nn.DataParallel(model)

    quadrat_preds = defaultdict(set)
    
    with torch.no_grad():
        for tiles, paths in tqdm(loader, desc="Inference"):
            tiles = tiles.to(device)
            
            # Forward
            logits = model(tiles)
            preds = torch.argmax(logits, dim=1)
            
            # Aggregate
            for i in range(len(paths)):
                path = paths[i]
                if path == "error_path": continue
                
                pred_idx = preds[i].item()
                predicted_species = class_names[pred_idx]
                
                filename = os.path.basename(path)
                quadrat_id = os.path.splitext(filename)[0]
                quadrat_preds[quadrat_id].add(predicted_species)
    
    final_preds = {k: sorted(list(v)) for k, v in quadrat_preds.items()}
    write_submission(final_preds, output_file)

# ==========================================
# MAIN
# ==========================================

def main():
    args = parse_args().parse_args()
    
    # Paths
    CLASSES_DIR = os.path.join(PLANT_HOME, args.classes_dir)
    TEST_DIR = os.path.join(PLANT_HOME, args.test_dir)
    MODEL_FILE = os.path.join(PLANT_HOME, args.model_path)
    BACKBONE_FILE = os.path.join(PLANT_HOME, args.backbone_path)
    
    # Settings
    GRID_SIZE = tuple(args.grid_size)
    IMAGE_SIZE = 518
    MODEL_CHECKPOINT = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count()
    BATCH_SIZE = args.batch_size * NUM_GPUS
    NUM_WORKERS = min(16, os.cpu_count())

    SUBMISSION_FILE = os.path.join(PLANT_HOME, "submissions", "lora"+str(GRID_SIZE[0])+"x"+str(GRID_SIZE[1])+".csv")


    print(f"Running Inference on {DEVICE} with {NUM_GPUS} GPUs")
    print(f"Grid Size: {GRID_SIZE}")
    print(f"Model: {MODEL_FILE}")

    # 1. Load Classes
    class_names = get_class_names(CLASSES_DIR)
    
    # 2. Data Loader
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = QuadratTilingDataset_Inference(
        TEST_DIR, 
        grid_size=GRID_SIZE, 
        transform=transform, 
        img_size=IMAGE_SIZE
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

    # 3. Load Model
    model = load_lora_model(
        MODEL_CHECKPOINT, 
        BACKBONE_FILE, 
        MODEL_FILE, 
        len(class_names), 
        DEVICE
    )

    # 4. Run
    run_inference(model, loader, class_names, DEVICE, SUBMISSION_FILE, NUM_GPUS)

if __name__ == "__main__":
    main()