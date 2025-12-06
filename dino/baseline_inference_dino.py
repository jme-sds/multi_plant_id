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
import argparse

# ==========================================
# CONFIGURATION & ARGS
# ==========================================
PLANT_HOME = os.getenv("PLANT_HOME")
assert PLANT_HOME is not None, "Please set home/root directory of PlantCLEF files to the environment variable PLANT_HOME."

def parse_args():
    parser = argparse.ArgumentParser(description="PlantCLEF Baseline Inference with Multi-Grid Pooling")
    
    # Grid Pooling: Pass pairs of integers. 
    # Example: --grids 2 2 3 3 4 4  -> Runs (2,2), (3,3), and (4,4)
    parser.add_argument("--grids", type=int, nargs='+', default=[2, 2, 3, 3], 
                        help="List of grid sizes to pool. Format: R1 C1 R2 C2 ... (default: 2x2 and 3x3)")
    
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--model_checkpoint", type=str, default="timm/vit_base_patch14_reg4_dinov2.lvd142m")
    
    # Paths
    parser.add_argument("--backbone_path", type=str, default="dinov2_model/model_best.pth.tar", help="Path to backbone weights")
    parser.add_argument("--fine_tuned_path", type=str, default="baseline_fine_tuned.pth", help="Path to fine-tuned classifier head (optional)")
    parser.add_argument("--classes_dir", type=str, default="images_max_side_800", help="Directory with class folders")
    parser.add_argument("--test_dir", type=str, default="quadrat/images", help="Directory with test images")
    parser.add_argument("--output", type=str, default="submission_baseline_pooled.csv", help="Output filename")
    
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

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.JPG", "*.JPEG"]
        for ext in extensions:
             img_paths.extend(glob.glob(os.path.join(self.data_dir, ext)))
        
        img_paths = sorted(img_paths)

        for img_path in img_paths:
            for i in range(self.num_tiles):
                self.samples.append((img_path, i))
                
        print(f"Grid {self.num_rows}x{self.num_cols}: Found {len(img_paths)} images -> {len(self.samples)} tiles.")

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
            dummy = torch.zeros((3, self.img_size, self.img_size))
            return dummy, "error"

    def _get_tile(self, img, tile_index):
        img_width, img_height = img.size
        
        tile_width = img_width // self.num_cols
        tile_height = img_height // self.num_rows
        
        row = tile_index // self.num_cols
        col = tile_index % self.num_cols
        
        left = col * tile_width
        top = row * tile_height
        
        right = img_width if col == self.num_cols - 1 else (col + 1) * tile_width
        bottom = img_height if row == self.num_rows - 1 else (row + 1) * tile_height
            
        return img.crop((left, top, right, bottom))

# ==========================================
# MODEL
# ==========================================

class SimpleClassifier(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        
        if hasattr(backbone, "embed_dim"):
            dim = backbone.embed_dim
        elif hasattr(backbone, "num_features"):
            dim = backbone.num_features
        else:
            dim = 768
            
        self.classifier = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        logits = self.classifier(cls_token)
        return logits

# ==========================================
# UTILS
# ==========================================

def get_class_names(root_dir):
    if not os.path.exists(root_dir):
        print(f"Warning: Class directory {root_dir} not found. Creating dummy classes.")
        return [str(i) for i in range(100)]
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"Loaded {len(classes)} classes.")
    return classes

def load_model(checkpoint_name, backbone_path, fine_tuned_path, num_classes, device):
    print(f"Creating model: {checkpoint_name}")
    base_model = timm.create_model(checkpoint_name, pretrained=True)
    
    # 1. Load Backbone (Challenge Weights)
    if os.path.exists(backbone_path):
        print(f"Loading backbone weights from {backbone_path}")
        checkpoint = torch.load(backbone_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        base_model.load_state_dict(state_dict, strict=False)
    else:
        print("Backbone path invalid, using default pretrained weights.")

    # 2. Attach Head
    model = SimpleClassifier(base_model, num_classes)
    
    # 3. Load Fine-Tuned Weights (If available)
    # This overwrites the backbone and the random head with your trained version
    if os.path.exists(fine_tuned_path):
        print(f"Loading fine-tuned classifier from {fine_tuned_path}")
        state = torch.load(fine_tuned_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        print(f"WARNING: Fine-tuned model not found at {fine_tuned_path}. Head is random!")
    
    model.to(device)
    model.eval()
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Parallelizing across {torch.cuda.device_count()} GPUs")
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
    args = parse_args().parse_args()
    
    # Setup Paths
    CLASSES_DIR = os.path.join(PLANT_HOME, args.classes_dir)
    TEST_DIR = os.path.join(PLANT_HOME, args.test_dir)
    BACKBONE_FILE = os.path.join(PLANT_HOME, args.backbone_path)
    FINE_TUNED_FILE = os.path.join(PLANT_HOME, args.fine_tuned_path)
    
    IMAGE_SIZE = 518
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = min(16, os.cpu_count())
    BATCH_SIZE = args.batch_size * torch.cuda.device_count()

    # Parse Grid Sizes
    # Convert [2, 2, 3, 3] -> [(2,2), (3,3)]
    if len(args.grids) % 2 != 0:
        raise ValueError("Grid argument must have an even number of integers (pairs of rows/cols).")
    
    grid_configs = []
    for i in range(0, len(args.grids), 2):
        grid_configs.append((args.grids[i], args.grids[i+1]))
    
    print(f"Pooling predictions from grids: {grid_configs}")

    # 1. Load Resources
    classes = get_class_names(CLASSES_DIR)
    model = load_model(args.model_checkpoint, BACKBONE_FILE, FINE_TUNED_FILE, len(classes), DEVICE)
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Inference Loop (Iterate over Grids)
    # We use a set to automatically handle duplicates (Union Pooling)
    pooled_preds = defaultdict(set) 
    
    for grid_size in grid_configs:
        print(f"\n--- Running Inference for Grid {grid_size} ---")
        
        dataset = QuadratTilingDataset_Inference(
            TEST_DIR, 
            grid_size=grid_size, 
            transform=transform,
            img_size=IMAGE_SIZE
        )
        
        if len(dataset) == 0:
            print("No images found for this grid. Skipping.")
            continue

        loader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            pin_memory=True
        )
        
        with torch.no_grad():
            for tiles, paths in tqdm(loader, desc=f"Grid {grid_size}"):
                tiles = tiles.to(DEVICE)
                
                # Forward
                logits = model(tiles)
                preds = torch.argmax(logits, dim=1)
                
                # Aggregate
                for i, path in enumerate(paths):
                    if path == "error": continue
                    
                    pred_idx = preds[i].item()
                    species_id = classes[pred_idx]
                    
                    filename = os.path.basename(path)
                    quadrat_id = os.path.splitext(filename)[0]
                    
                    # Add to pooled set
                    pooled_preds[quadrat_id].add(species_id)

    # 3. Write Final Pooled Submission
    final_preds = {k: sorted(list(v)) for k, v in pooled_preds.items()}
    write_submission(final_preds, args.output)

if __name__ == "__main__":
    main()