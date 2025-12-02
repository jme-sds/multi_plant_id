import os
import sys

# --- 1. CRITICAL SETUP ---
PLANT_HOME = "/scratch/ezq9qu"
os.environ["PLANT_HOME"] = PLANT_HOME

print(f"DEBUG: PLANT_HOME is configured as: '{PLANT_HOME}'")

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import faiss
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
import csv
from torchvision import transforms
from tqdm import tqdm

# Try importing PEFT for LoRA
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: 'peft' library not found. LoRA training will not work. (pip install peft)")

try:
    from data_loader import SinglePlantDataLoader
    from quadrat import QuadratTilingDataset_Inference
    import knn as knn_utils 
except ImportError:
    from loading.data_loader import SinglePlantDataLoader
    from loading.quadrat import QuadratTilingDataset_Inference
    import knn as knn_utils

# --- Configuration ---
BATCH_SIZE = 32
NUM_WORKERS = 4 

class ResNetWrapper(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
    def forward_features(self, x):
        return self.model(x)

def get_model(model_name, device, num_classes=None, use_lora=False):
    """
    Loads model. If num_classes is provided, attaches a classification head
    (required for training). If use_lora is True, wraps in PEFT.
    """
    print(f"Loading model: {model_name}...")
    
    # 1. Load Base Model
    if "resnet" in model_name.lower():
        # For training, we need num_classes support, so we don't use the Wrapper if training
        if num_classes:
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        else:
            model = ResNetWrapper(model_name, pretrained=True)
            
    elif "dino" in model_name.lower():
        # DINOv2 logic
        if num_classes:
            # Load with head for training
            model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=num_classes)
        else:
            # Load without head for extraction
            checkpoint_path = os.path.join(PLANT_HOME, "dinov2_model/model_best.pth.tar")
            if os.path.exists(checkpoint_path):
                print(f"Loading local DINOv2 checkpoint from {checkpoint_path}")
                try:
                    model = knn_utils.load_model(device)
                except AttributeError:
                    model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True)
            else:
                model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True)

    model.to(device)

    # 2. Apply LoRA if requested
    if use_lora and num_classes:
        if not PEFT_AVAILABLE:
            raise ImportError("Cannot use LoRA because 'peft' is not installed.")
        
        print("Applying LoRA adapters...")
        # Config for ViT (DINOv2)
        target_modules = ["qkv"] if "dino" in model_name else ["conv1", "conv2"] # Simple guess for ResNet
        
        peft_config = LoraConfig(
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=target_modules 
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model

def train_lora_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
    return running_loss / len(loader)

def build_faiss_index(embs, device):
    faiss_dir = os.path.join(PLANT_HOME, "knn", "faiss_index")
    os.makedirs(faiss_dir, exist_ok=True)
    faiss_file = os.path.join(faiss_dir, "faiss.idx")

    use_gpu = torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
    input_dim = embs.shape[1]
    
    if os.path.exists(faiss_file):
        try:
            index = faiss.read_index(faiss_file)
            if index.d != input_dim:
                print(f"Index dim mismatch ({index.d} vs {input_dim}). Rebuilding...")
            else:
                if use_gpu:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                return index
        except:
            pass

    index = faiss.IndexFlatIP(input_dim)
    if use_gpu:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(embs.astype("float32"))
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), faiss_file)
        return gpu_index
    else:
        index.add(embs.astype("float32"))
        faiss.write_index(index, faiss_file)
        return index

def pool_and_aggregate(all_grid_results, min_votes=1):
    quadrat_vote_pool = defaultdict(list)
    print("\nPooling votes from all grids...")
    for grid_idx, (preds, paths) in enumerate(all_grid_results):
        for tile_neighbors, tile_path in zip(preds, paths):
            quadrat_id = os.path.splitext(os.path.basename(tile_path))[0]
            if isinstance(tile_neighbors, np.ndarray):
                votes = tile_neighbors.flatten().tolist()
            else:
                votes = list(tile_neighbors)
            quadrat_vote_pool[quadrat_id].extend(votes)

    print(f"  Aggregating votes for {len(quadrat_vote_pool)} unique quadrats...")
    final_predictions = {}
    for q_id, all_votes in quadrat_vote_pool.items():
        counter = Counter(all_votes)
        species_kept = [sp for sp, count in counter.items() if count >= min_votes]
        final_predictions[q_id] = sorted(species_kept)
    return final_predictions

def main():
    parser = argparse.ArgumentParser(description="PlantCLEF 2025 Pipeline")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "dinov2"])
    parser.add_argument("--quadrat_dir", type=str, required=True)
    parser.add_argument("--single_plant_dir", type=str, required=True)
    parser.add_argument("--grids", type=int, nargs='+', default=[4, 5, 6])
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--votes", type=int, default=3)
    
    # --- LoRA Training Args ---
    parser.add_argument("--train_lora", action="store_true", help="Enable LoRA training on Single Plant data before inference")
    parser.add_argument("--epochs", type=int, default=1, help="Number of LoRA training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for LoRA")
    
    args = parser.parse_args()

    # Dynamic Image Size
    if "dino" in args.backbone.lower():
        IMG_SIZE = 518
    else:
        IMG_SIZE = 224
        
    print(f"Selected Backbone: {args.backbone}. Setting IMG_SIZE to {IMG_SIZE}.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. SETUP DATA ---
    print("\n--- Setting up Single Plant Dataloader ---")
    single_loader_wrapper = SinglePlantDataLoader(
        data_dir=args.single_plant_dir,
        resize_size=IMG_SIZE + 32,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    train_loader, val_loader, _ = single_loader_wrapper.get_dataloaders()
    train_dataset = train_loader.dataset.dataset 
    class_to_speciesid = {i: int(cls_name) for i, cls_name in enumerate(train_dataset.classes)}
    knn_utils.class_to_speciesid = class_to_speciesid 
    num_classes = len(class_to_speciesid)

    # --- 2. OPTIONAL LORA TRAINING ---
    lora_checkpoint_path = os.path.join(PLANT_HOME, f"lora_{args.backbone}_best.pth")

    if args.train_lora:
        print("\n--- STARTING LORA TRAINING ---")
        # Initialize model WITH HEAD for training
        model = get_model(args.backbone, device, num_classes=num_classes, use_lora=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(1, args.epochs + 1):
            loss = train_lora_epoch(model, train_loader, optimizer, criterion, device, epoch)
            print(f"Epoch {epoch} finished. Loss: {loss:.4f}")
            
            # Simple save (you could add validation logic here)
            torch.save(model.state_dict(), lora_checkpoint_path)
            
        print(f"LoRA training complete. Saved to {lora_checkpoint_path}")
        
        # Cleanup to free memory before inference
        del model
        torch.cuda.empty_cache()
        
        # Force re-extraction of embeddings since model changed
        FORCE_REEXTRACT = True
    else:
        FORCE_REEXTRACT = False

    # --- 3. LOAD MODEL FOR INFERENCE ---
    print("\n--- Loading Model for Inference ---")
    
    # If we trained LoRA, we need to load those weights. 
    # But for k-NN extraction, we usually want the BACKBONE features, NOT the class logits.
    # However, 'timm' models with LoRA are tricky to strip the head off dynamically.
    # Strategy: Load model with head + LoRA, then use forward_features (which usually ignores head).
    
    if args.train_lora or os.path.exists(lora_checkpoint_path):
        print("Loading model with trained LoRA adapter...")
        # Load same structure as training
        model = get_model(args.backbone, device, num_classes=num_classes, use_lora=True)
        # Load weights
        if os.path.exists(lora_checkpoint_path):
            state_dict = torch.load(lora_checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
    else:
        # Standard frozen backbone
        model = get_model(args.backbone, device, num_classes=None, use_lora=False)
    
    model.eval()

    # --- 4. EXTRACT EMBEDDINGS (MEMORY BANK) ---
    emb_dir = os.path.join(PLANT_HOME, "knn", "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    
    # Naming the file differently if LoRA was used
    suffix = "_lora" if (args.train_lora or os.path.exists(lora_checkpoint_path)) else ""
    train_emb_file = os.path.join(emb_dir, f"train_embs_{args.backbone}{suffix}.npz")
    
    if os.path.exists(train_emb_file) and not FORCE_REEXTRACT:
        print(f"Loading cached training embeddings from {train_emb_file}")
        with np.load(train_emb_file) as data:
            train_embs = data['embs']
            train_labels = data['labels']
    else:
        print("Extracting training embeddings (Model changed or cache missing)...")
        train_embs, train_labels = knn_utils.extract_embeddings(train_loader, train_dataset, model, device)
        np.savez(train_emb_file, embs=train_embs, labels=train_labels)

    # --- 5. BUILD INDEX ---
    print("Normalizing and building Index...")
    faiss.normalize_L2(train_embs)
    index = build_faiss_index(train_embs, device)

    # --- 6. QUADRAT INFERENCE ---
    all_grid_results = [] 
    pil_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for grid_n in args.grids:
        print(f"\n--- Processing Grid Size: {grid_n}x{grid_n} ---")
        quadrat_dataset = QuadratTilingDataset_Inference(
            data_dir=args.quadrat_dir, grid_size=(grid_n, grid_n), transform=pil_transform
        )
        quadrat_loader = DataLoader(
            quadrat_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        )

        quad_emb_file = os.path.join(emb_dir, f"quadrat_embs_{args.backbone}{suffix}_{grid_n}x{grid_n}.npz")
        
        if os.path.exists(quad_emb_file) and not FORCE_REEXTRACT:
            print(f"Loading cached embeddings: {quad_emb_file}")
            with np.load(quad_emb_file) as data:
                quadrat_embs = data['embs']
                quadrat_paths = data['paths']
        else:
            print("Extracting embeddings...")
            quadrat_embs, quadrat_paths = knn_utils.extract_unlabeled_embeddings(quadrat_loader, model, device)
            np.savez(quad_emb_file, embs=quadrat_embs, paths=quadrat_paths)
            
        print("Running KNN...")
        faiss.normalize_L2(quadrat_embs)
        tile_preds = knn_utils.knn_predict(quadrat_embs, index=index, faiss_labels=train_labels, k=args.neighbors)
        all_grid_results.append((tile_preds, quadrat_paths))

    final_preds = pool_and_aggregate(all_grid_results, min_votes=args.votes)

    out_csv = os.path.join(PLANT_HOME, "submissions", f"submission_{args.backbone}{suffix}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    knn_utils.write_submission(final_preds, out_csv=out_csv)
    print(f"\nDone! Saved to: {out_csv}")

if __name__ == "__main__":
    main()