import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import random

# ==========================================
# CONFIGURATION
# ==========================================
# Paths
PLANT_HOME = "/scratch/jme3qd/data/plantclef2025" # Base directory for models
MODEL_CHECKPOINT = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
#BACKBONE_PATH = os.path.join(PLANT_HOME, "dinov2_model/model_best.pth.tar")
BACKBONE_PATH = "baseline_fine_tuned.pth"
LUCAS_PATH = os.path.join(PLANT_HOME, "lucas") # Folder containing unlabelled .jpg images
PLANTCLEF_PATH = os.path.join(PLANT_HOME, "images_max_side_800") # Folder with subfolders for each species
TEST_IMAGES_PATH = os.path.join(PLANT_HOME,"quadrat/images")

# Training Config
# A100 Optimization:
# - Batch Size 1024 (256 per GPU). 
# - If you still have VRAM headroom, you can try 2048.
BATCH_SIZE = 1024    
IMAGE_SIZE = 518
EPOCHS = 15         
LEARNING_RATE = 0.01 
MAX_SAMPLES = None  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()

print(f"Running on {DEVICE} with {NUM_GPUS} GPUs.")

# ==========================================
# DATASET
# ==========================================

class PlantClefSupervisedDataset(Dataset):
    def __init__(self, root_dir, split='train', max_samples=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        if os.path.exists(root_dir):
            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                if not os.path.isdir(cls_folder): continue
                for f in os.listdir(cls_folder):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append((os.path.join(cls_folder, f), self.class_to_idx[cls_name]))
        
        if max_samples is not None and len(self.samples) > max_samples:
            print(f"Limiting dataset to {max_samples} samples.")
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), torch.tensor(0, dtype=torch.long)

# ==========================================
# MODEL WRAPPER
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
    
    def forward(self, x, labels=None):
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        logits = self.classifier(cls_token)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

# ==========================================
# TRAINING LOOP
# ==========================================

def main():
    # 1. Load Data
    print("Initializing Dataset...")
    dataset = PlantClefSupervisedDataset(PLANTCLEF_PATH, max_samples=MAX_SAMPLES)
    if len(dataset) == 0:
        print("No training data found! Check paths.")
        return
        
    print(f"Training on {len(dataset)} images across {len(dataset.classes)} classes.")
    
    # DATALOADER OPTIMIZATIONS
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,          # 16 workers fits in 100GB RAM safely with batch size 1024
        pin_memory=True,         # Critical for GPU throughput
        persistent_workers=True, # Keeps workers alive, reduces CPU overhead between epochs
        prefetch_factor=2        # Buffers 2 batches per worker
    )

    # 2. Load Backbone
    print("Loading Backbone...")
    base_model = timm.create_model(MODEL_CHECKPOINT, pretrained=True)
    
    if os.path.exists(BACKBONE_PATH):
        print(f"Loading organizer weights from {BACKBONE_PATH}")
        checkpoint = torch.load(BACKBONE_PATH, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        base_model.load_state_dict(state_dict, strict=False)
    else:
        print("Organizer checkpoint not found. Using default DINOv2 weights.")

    # Freeze Backbone 
    for param in base_model.parameters():
        param.requires_grad = False
    
    # 3. Attach Head
    model = SimpleClassifier(base_model, len(dataset.classes))
    model.to(DEVICE)
    
    # 4. Multi-GPU
    if NUM_GPUS > 1:
        print(f"Wrapping model in DataParallel for {NUM_GPUS} GPUs.")
        model = nn.DataParallel(model)

    if NUM_GPUS > 1:
        head_params = model.module.classifier.parameters()
    else:
        head_params = model.classifier.parameters()
        
    optimizer = optim.SGD(head_params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Scaler for AMP (Automatic Mixed Precision)
    scaler = torch.amp.GradScaler('cuda')

    # 6. Train
    print("\nStarting Linear Probing...")
    model.train() 
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Use AMP for A100 optimization
            with torch.amp.autocast('cuda'):
                if NUM_GPUS > 1:
                    loss_vec, logits = model(imgs, labels)
                    loss = loss_vec.mean()
                else:
                    loss, logits = model(imgs, labels)
            
            # Scaler steps
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.2%}"})
        
        scheduler.step()
        print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(dataloader):.4f}, Acc: {correct/total:.2%}")

    # 7. Save
    print("Saving model...")
    if NUM_GPUS > 1:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    torch.save(state_dict, "baseline_fine_tuned_30epoch.pth")
    print("Saved to baseline_fine_tuned.pth")

if __name__ == "__main__":
    main()
