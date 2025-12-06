import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from PIL import Image
from tqdm import tqdm
import numpy as np
import timm
import random
import glob
import csv
from collections import defaultdict
import argparse


# Default to current directory if env var not set to avoid immediate crash
PLANT_HOME = os.getenv("PLANT_HOME")
assert PLANT_HOME is not None, "Please set home/root directory of PlantCLEF files to the environment variable PLANT_HOME."

# ==========================================
# DATASETS
# ==========================================

class LucasSSLDataset(Dataset):
    def __init__(self, root_dir, processor, max_samples=None):
        self.root_dir = root_dir
        self.files = []
        if os.path.exists(root_dir):
            for root, dirs, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.files.append(os.path.join(root, filename))
            if max_samples is not None and len(self.files) > max_samples:
                print(f"Limiting SSL dataset to {max_samples} samples (randomly selected).")
                random.shuffle(self.files)
                self.files = self.files[:max_samples]
        self.processor = processor
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert("RGB")
        view1 = self.augment(image)
        view2 = self.augment(image)
        return view1, view2

class PlantClefSupervisedDataset(Dataset):
    def __init__(self, root_dir, processor, split='train', max_samples=None):
        self.root_dir = root_dir
        self.processor = processor
        if os.path.exists(root_dir):
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.samples = []
            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                if not os.path.isdir(cls_folder): continue
                for f in os.listdir(cls_folder):
                    if f.lower().endswith(('.jpg', '.png')):
                        self.samples.append((os.path.join(cls_folder, f), self.class_to_idx[cls_name]))
        else:
            self.classes = []
            self.samples = []
            
        if max_samples is not None and len(self.samples) > max_samples:
            print(f"Limiting Supervised dataset to {max_samples} samples (randomly selected).")
            random.shuffle(self.samples) 
            self.samples = self.samples[:max_samples]
            
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return {"pixel_values": image, "labels": torch.tensor(label, dtype=torch.long)}

class QuadratTilingDataset_Inference(Dataset):
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
          dummy_tile = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)) 
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
# MODELS & LOSSES
# ==========================================

class SimCLRHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat((z_i, z_j), dim=0)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)
        loss = self.criterion(sim_matrix, labels)
        return loss / (2 * batch_size)

# ==========================================
# WRAPPERS FOR PARALLEL EXECUTION
# ==========================================

class SSLBackboneWrapper(nn.Module):
    """
    Wraps the LoRA backbone to ensure forward() returns the embeddings
    needed for SimCLR, enabling nn.DataParallel to work correctly.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
    def forward(self, x):
        # We need the embeddings, not the class logits
        # timm returns features [Batch, Tokens, Dim]
        features = self.backbone.forward_features(x)
        # Return CLS token (Index 0)
        return features[:, 0]

class LoRAClassifier(nn.Module):
    """
    Wraps backbone + classifier for supervised training.
    """
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        dim = backbone.base_model.model.embed_dim if hasattr(backbone, 'base_model') else backbone.embed_dim
        self.classifier = nn.Linear(dim, n_classes)
    
    def forward(self, x, labels=None):
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        logits = self.classifier(cls_token)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            # Return tuple for DataParallel compatibility
            return loss, logits 
        return logits

# ==========================================
# UTILITIES
# ==========================================

def get_base_model_with_lora():
    print("Loading base model...")
    model = timm.create_model(MODEL_CHECKPOINT, pretrained=True)
    
    checkpoint_path = os.path.join(PLANT_HOME, "dinov2_model/model_best.pth.tar")
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False) 
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Checkpoint not found at {checkpoint_path}, using downloaded pretrained weights.")

    model.eval() 
    for p in model.parameters():
        p.requires_grad = False

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["qkv"], 
        lora_dropout=0.1,
        bias="none",
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def train_ssl(model, dataloader, epochs):
    print("\n--- Starting Stage 1: SSL on Pseudo Quadrats (LUCAS) ---")
    
    model.to(DEVICE)
    model.train()
    
    # 1. Wrap model to return embeddings for DataParallel
    ssl_model = SSLBackboneWrapper(model).to(DEVICE)
    
    # 2. Projection Head
    hidden_size = model.base_model.model.embed_dim if hasattr(model.base_model, 'model') else model.embed_dim
    projection_head = SimCLRHead(hidden_size).to(DEVICE)
    
    # 3. Apply DataParallel if multiple GPUs
    if NUM_GPUS > 1:
        print(f"Wrapping SSL models in DataParallel on {NUM_GPUS} GPUs...")
        ssl_model = nn.DataParallel(ssl_model)
        projection_head = nn.DataParallel(projection_head)
    
    optimizer = optim.AdamW([
        {'params': model.parameters()}, 
        {'params': projection_head.parameters()}
    ], lr=LEARNING_RATE)
    
    criterion = NTXentLoss()

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"SSL Epoch {epoch+1}/{epochs}")
        
        for view1, view2 in pbar:
            view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass (Parallelized)
            # Returns CLS tokens [Batch, Dim]
            feat1 = ssl_model(view1)
            feat2 = ssl_model(view2)
            
            # Projection (Parallelized)
            z1 = projection_head(feat1)
            z2 = projection_head(feat2)
            
            z1 = nn.functional.normalize(z1, dim=1)
            z2 = nn.functional.normalize(z2, dim=1)
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")
    
    # Unwrap model before returning to avoid issues later
    if NUM_GPUS > 1:
        return ssl_model.module.backbone
    return ssl_model.backbone

def train_supervised(model, dataloader, num_classes, epochs):
    print("\n--- Starting Stage 2: Supervised Training on Single Plants ---")
    
    # Wrap in Classifier
    classifier_model = LoRAClassifier(model, num_classes).to(DEVICE)
    
    # Apply DataParallel
    if NUM_GPUS > 1:
        print(f"Wrapping Supervised model in DataParallel on {NUM_GPUS} GPUs...")
        parallel_classifier = nn.DataParallel(classifier_model)
    else:
        parallel_classifier = classifier_model
    
    optimizer = optim.AdamW(parallel_classifier.parameters(), lr=LEARNING_RATE)
    parallel_classifier.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            imgs = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass (Parallelized)
            # Returns tuple (loss, logits) but loss is a vector of losses per-GPU
            results = parallel_classifier(imgs, labels)
            
            # DataParallel returns loss as a vector [batch_size/ngpus] size if reduction='none'
            # or [ngpus] size if reduction='mean' (default cross entropy usually reduces)
            # However, since we return (loss, logits), DP gathers them. 
            # Loss will be a tensor of shape [NUM_GPUS]. We take mean.
            
            loss_vec, logits = results
            loss = loss_vec.mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'acc': correct/total})
            
    # Return the unwrapped model for saving/inference
    if NUM_GPUS > 1:
        return parallel_classifier.module
    return classifier_model

def evaluate(model, dataloader):
    model.eval()
    
    # Wrap for evaluation parallelization
    if NUM_GPUS > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    correct = 0
    total = 0
    print("\nEvaluating...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            if hasattr(model, 'module') and hasattr(model.module, 'classifier'):
                logits = model(imgs) # LoRA Forward handles return
            elif hasattr(model, 'classifier'):
                logits = model(imgs)
            else: 
                # Baseline
                logits = model(imgs)
                if isinstance(logits, tuple): logits = logits[1] # Handle (loss, logits) if accidentally used
                
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = correct / total
    return acc

def write_submission(pred_dict, out_csv="submission.csv"):
    print("Writing submission to {}".format(out_csv))
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quadrat_id", "species_ids"])
        for quad, species in pred_dict.items():
            s = "[" + ", ".join(str(x) for x in species) + "]"
            writer.writerow([quad, s])

def run_inference_and_submission(model, test_loader, class_names, output_filename):
    print(f"\n--- Running Inference for {output_filename} ---")
    model.eval()
    
    # Wrap for inference parallelization
    if NUM_GPUS > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    quadrat_preds = defaultdict(set)
    
    with torch.no_grad():
        for tiles, paths in tqdm(test_loader, desc="Inference"):
            tiles = tiles.to(DEVICE)
            
            logits = model(tiles)
            preds = torch.argmax(logits, dim=1)
            
            for i in range(len(paths)):
                path = paths[i]
                pred_idx = preds[i].item()
                predicted_species = class_names[pred_idx]
                filename = os.path.basename(path)
                quadrat_id = os.path.splitext(filename)[0]
                quadrat_preds[quadrat_id].add(predicted_species)
    
    final_preds = {k: sorted(list(v)) for k, v in quadrat_preds.items()}
    write_submission(final_preds, output_filename)

def parse_args():
    parser = argparse.ArgumentParser(description="PlantCLEF Training Pipeline")
    parser.add_argument("--ssl_epochs", type=int, default=5, help="Number of SSL epochs")
    parser.add_argument("--sft_epochs", type=int, default=15, help="Number of Supervised Fine-Tuning epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size_per_gpu", type=int, default=128, help="Batch size per GPU, 64 for A6000, 128 for A100 80gb")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples for datasets (debug)")
    parser.add_argument("--lora_path", type=str, default="lora_lucas_ssl_weights", help="Name of LoRA weights folder")
    parser.add_argument("--grid_size", type=int, nargs=2, default=(3, 4), help="Grid size for inference")
    parser.add_argument("--lora_save_path", type=str, default="final_lora_classifier", help="Name of LoRA weights folder")
    return parser

def main():
    args = parse_args().parse_args()

    # --- SETUP VARS ---
    # Construct paths dynamically
    LUCAS_PATH = os.path.join(PLANT_HOME, "lucas/images") 
    PLANTCLEF_PATH = os.path.join(PLANT_HOME, "images_max_side_800")
    TEST_IMAGES_PATH = os.path.join(PLANT_HOME,"quadrat/images")
    
    # LORA_PATH is the FOLDER where weights are saved/loaded from
    LORA_PATH = os.path.join(PLANT_HOME, "lucas/", args.lora_path)
    
    # FINAL MODEL is the FILE where the classifier + adapters are saved
    MODEL_PATH = os.path.join(PLANT_HOME, "lucas/models/", args.lora_save_path,"_model.pth")
    
    global MODEL_CHECKPOINT, IMAGE_SIZE, SSL_EPOCHS, SFT_EPOCHS, LEARNING_RATE, GRID_SIZE, MAX_SAMPLES, DEVICE, NUM_GPUS, NUM_CPUS, BATCH_SIZE, NUM_WORKERS
    
    MODEL_CHECKPOINT = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
    IMAGE_SIZE = 518    
    SSL_EPOCHS = args.ssl_epochs     
    SFT_EPOCHS = args.sft_epochs
    LEARNING_RATE = args.learning_rate
    GRID_SIZE = tuple(args.grid_size)
    MAX_SAMPLES = args.max_samples
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count()
    NUM_CPUS = os.cpu_count()
    BATCH_SIZE = args.batch_size_per_gpu * NUM_GPUS
    NUM_WORKERS = min(16, NUM_CPUS) # Cap workers

    print(f"Running on device: {DEVICE}")
    print(f"Number of GPUs available: {NUM_GPUS}")
    if NUM_GPUS > 1:
        print("Multi-GPU DataParallel Mode Enabled.")

    # --- LOAD DATASETS ---
    try:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    except:
        print("Could not load HF processor, using default.")
        processor = None

    lucas_ds = LucasSSLDataset(LUCAS_PATH, processor, max_samples=MAX_SAMPLES)
    # Only create loader if we need SSL
    lucas_loader = DataLoader(lucas_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    plant_ds = PlantClefSupervisedDataset(PLANTCLEF_PATH, processor, split='train', max_samples=MAX_SAMPLES)
    plant_loader = DataLoader(plant_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    num_classes = len(plant_ds.classes)
    class_names = plant_ds.classes
    print(f"Found {num_classes} classes.")

    inference_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    test_ds = QuadratTilingDataset_Inference(TEST_IMAGES_PATH, grid_size=GRID_SIZE, transform=inference_transform)
    test_loader = None
    if len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- MODEL A: LoRA + SSL + SFT ---
    print("\n=== Model A: LoRA Pre-trained on LUCAS ===")
    
    # 1. Initialize Base Model + Random LoRA Adapters
    model_lora = get_base_model_with_lora()

    # 2. Check for existing SSL weights (Adapters only)
    if os.path.exists(LORA_PATH):
        print(f"Found existing SSL adapters at {LORA_PATH}")
        print("Loading adapters into backbone...")
        # Load adapters into the PEFT model
        model_lora.load_adapter(LORA_PATH, adapter_name="default")
    else:
        print("No existing SSL adapters found. Starting SimCLR training...")
        # Train SSL (updates adapters)
        model_lora = train_ssl(model_lora, lucas_loader, SSL_EPOCHS)
        
        # Save Adapters (Folder)
        print(f"Saving SSL adapters to {LORA_PATH}")
        model_lora.save_pretrained(LORA_PATH)
    
    # 3. Supervised Fine-Tuning (SFT)
    # train_supervised will wrap the model in LoRAClassifier (adding the head)
    # and train both the head and the adapters (or just head, depending on config)
    final_model_a = train_supervised(model_lora, plant_loader, num_classes, SFT_EPOCHS)
    
    # 4. Save Final Full Model (State Dict of Classifier + Adapters)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(final_model_a.state_dict(), MODEL_PATH)
    print(f"Saved final fine-tuned model to {MODEL_PATH}")
    
    # 5. Inference
    if test_loader:
        run_inference_and_submission(final_model_a, test_loader, class_names, "submission_model_lora_"+str(args.grid_size[0])+"x"+str(args.grid_size[1])+"_.csv")
if __name__ == "__main__":
    main()
