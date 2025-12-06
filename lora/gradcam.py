import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm
from peft import PeftModel, LoraConfig, get_peft_model
import numpy as np
import cv2
import csv
import random
import glob
import argparse


PLANT_HOME = os.getenv("PLANT_HOME")
assert PLANT_HOME is not None, "Please set home/root directory of PlantCLEF files to the environment variable PLANT_HOME."

# ==========================================
# GRADCAM CLASS
# ==========================================

class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = None
        
        # Recursively search for the last 'blocks' attribute in the backbone
        candidates = [model]
        if hasattr(model, 'backbone'): candidates.append(model.backbone)
        if hasattr(model.backbone, 'base_model'): candidates.append(model.backbone.base_model)
        if hasattr(model.backbone, 'base_model') and hasattr(model.backbone.base_model, 'model'): candidates.append(model.backbone.base_model.model)
        
        for candidate in candidates:
            if hasattr(candidate, 'blocks'):
                # CRITICAL CHANGE: Hook 'norm1' inside the last block
                # Hooking the block output yields zero gradients for patches because
                # the classifier only uses the CLS token (index 0) from the block output.
                # Hooking norm1 (input to Attention) captures the gradients BEFORE 
                # they are mixed into the CLS token via Attention.
                self.target_layer = candidate.blocks[-1].norm1
                break
        
        if self.target_layer is None:
            raise ValueError("Could not find 'blocks[-1].norm1' in model hierarchy. Check model structure.")
        
        print(f"Hooked target layer: {self.target_layer}")

        self.target_layer.register_forward_hook(self.save_activation_and_hook_grad)

    def save_activation_and_hook_grad(self, module, input, output):
        self.activations = output
        output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def generate_cam(self, input_tensor, target_class_idx):
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        score = output[0, target_class_idx]
        score.backward()
        
        grads = self.gradients
        acts = self.activations
        
        if grads is None or acts is None:
            print("Error: No gradients captured.")
            return np.zeros((14, 14)) 

        # Handle Tokens
        num_patches = (IMAGE_SIZE // 14) ** 2
        offset = acts.shape[1] - num_patches
        
        grads = grads[:, offset:, :]
        acts = acts[:, offset:, :]
        
        weights = torch.mean(grads, dim=1, keepdim=True) 
        cam = torch.sum(weights * acts, dim=2) 
        
        grid_dim = int(np.sqrt(num_patches))
        cam = cam.reshape(1, grid_dim, grid_dim)
        
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()[0]
        cam = np.nan_to_num(cam)
        
        # Debug: Print raw range to ensure gradients are flowing
        raw_max = np.max(cam)
        if raw_max == 0:
            print(f"DEBUG: Zero gradients for class {target_class_idx}. Check model/weights.")
        
        # Contrast Stretching (Clip outliers)
        percentile_99 = np.percentile(cam, 99)
        percentile_1 = np.percentile(cam, 1)
        cam = np.clip(cam, percentile_1, percentile_99)
        
        # Normalize 0-1
        max_val = np.max(cam)
        min_val = np.min(cam)
        
        if max_val - min_val > 1e-7:
            cam = (cam - min_val) / (max_val - min_val)
        else:
            cam = np.zeros_like(cam) 
        
        # Contrast Boost
        cam = np.power(cam, 2) 
            
        return cam

# ==========================================
# MODEL DEFINITIONS
# ==========================================

class LoRAClassifier(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        if hasattr(backbone, "base_model") and hasattr(backbone.base_model, "model"):
             dim = backbone.base_model.model.embed_dim
        elif hasattr(backbone, "embed_dim"):
             dim = backbone.embed_dim
        else:
             dim = 768 
        self.classifier = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        logits = self.classifier(cls_token)
        return logits

# ==========================================
# UTILITIES
# ==========================================

def load_species_map(csv_path):
    mapping = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Species map CSV not found at {csv_path}. Using IDs.")
        return mapping
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("species_id", "").strip()
                name = row.get("species", "").strip()
                if sid and name:
                    mapping[sid] = name
        print(f"Loaded {len(mapping)} species names.")
    except Exception as e:
        print(f"Error reading species CSV: {e}")
    return mapping

def load_class_names(root_dir):
    if os.path.exists(root_dir):
        return sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    return [f"Species_{i}" for i in range(NUM_CLASSES)]

def get_random_image(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Quadrat directory not found: {directory}")
    files = glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.jpeg"))
    if not files:
        raise FileNotFoundError(f"No jpg images found in {directory}")
    choice = random.choice(files)
    print(f"Selected random image: {choice}")
    return choice

def load_model(device, num_classes):
    print(f"Loading {MODEL_TYPE} model...")
    base_model = timm.create_model(MODEL_CHECKPOINT, pretrained=True)
    
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="none",
    )
    base_model = get_peft_model(base_model, config)
    model = LoRAClassifier(base_model, num_classes)
    
    if os.path.exists(FINE_TUNED_MODEL_PATH):
        print(f"Loading full model state from {FINE_TUNED_MODEL_PATH}")
        full_state = torch.load(FINE_TUNED_MODEL_PATH, map_location=device)
        keys = model.load_state_dict(full_state, strict=False)
        print(f"Weights loaded. Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)}")
    else:
        print(f"WARNING: Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}")

    # CRITICAL FIX FOR GRADCAM: Unfreeze parameters
    # Even though we are in eval mode, we need PyTorch to track gradients
    # through the backbone to the input image for GradCAM to work.
    for param in model.parameters():
        param.requires_grad = True

    model.to(device)
    model.eval() 
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

# ==========================================
# VISUALIZATION
# ==========================================

def apply_heatmap_standard(image_pil, cam_mask):
    """
    Standard Heatmap Overlay (Jet Colormap)
    Guarantees visibility by blending 50/50 with original image.
    """
    w, h = image_pil.size
    
    # 1. Resize mask
    cam_mask_resized = cv2.resize(cam_mask, (w, h))
    
    # 2. Colorize (Blue=Low, Red=High)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_mask_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap_rgb)
    
    # 3. Blend 50% opacity
    result = Image.blend(image_pil.convert("RGB"), heatmap_pil, alpha=0.5)
    return result

def visualize_gradcam(image_path, model, class_ids, species_map, device):
    if image_path is None:
        try:
            image_path = get_random_image(QUADRAT_DIR)
        except Exception as e:
            print(e)
            return

    gradcam = ViTGradCAM(model)
    original_img = Image.open(image_path).convert("RGB")
    final_canvas = original_img.copy()
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    if SINGLE_PLANT_MODE:
        print("Running in SINGLE PLANT MODE (No Tiling)...")
        rows, cols = 1, 1
    else:
        print(f"Running in QUADRAT MODE (Tiling {GRID_SIZE})...")
        rows, cols = GRID_SIZE

    img_width, img_height = original_img.size
    tile_width = img_width // cols
    tile_height = img_height // rows
    
    transform = get_transforms()
    
    print(f"Generating GradCAM for {image_path}...")
    
    for row in range(rows):
        for col in range(cols):
            left = col * tile_width
            top = row * tile_height
            right = (col + 1) * tile_width if col != cols - 1 else img_width
            bottom = (row + 1) * tile_height if row != rows - 1 else img_height
            
            tile = original_img.crop((left, top, right, bottom))
            input_tensor = transform(tile).unsqueeze(0).to(device)
            input_tensor.requires_grad = True 
            
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            class_idx = pred_idx.item()
            
            species_id = class_ids[class_idx] 
            display_name = species_map.get(str(species_id), species_id) 
            
            cam_map = gradcam.generate_cam(input_tensor, class_idx)
            
            # Use Standard Overlay (Guaranteed visibility)
            heatmap_tile = apply_heatmap_standard(tile, cam_map)
            final_canvas.paste(heatmap_tile, (left, top))
            
            draw = ImageDraw.Draw(final_canvas) 
            draw.rectangle([left, top, right, bottom], outline="white", width=3)
            
            if len(display_name) > 25:
                display_name = display_name[:22] + "..."
                
            label_text = f"{display_name}\n{conf.item():.2f}"
            text_pos = (left + 10, top + 10)
            bbox = draw.textbbox(text_pos, label_text, font=font)
            draw.rectangle(bbox, fill="black")
            draw.text(text_pos, label_text, fill="white", font=font)

    final_canvas.save(OUTPUT_PATH)
    print(f"GradCAM visualization saved to {OUTPUT_PATH}")

def parse_args():
    parser = argparse.ArgumentParser(description="GradCAM Visualization")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the image for GradCAM visualization")
    parser.add_argument("--output_path", type=str, default="./gradcam.jpg", help="Path to save the GradCAM visualization")
    parser.add_argument("--grid_size", type=int, nargs=2, default=(3, 3), help="Grid size for GradCAM visualization")
    parser.add_argument("--single_plant_mode",type=bool, default=False, help="Run in SINGLE PLANT MODE (No Tiling)")
    return parser

def main():
    args = parse_args().parse_args()
    global OUTPUT_PATH, MODEL_TYPE, MODEL_CHECKPOINT, QUADRAT_DIR, SPECIES_MAP_FILE, CLASS_LIST_DIR, FINE_TUNED_MODEL_PATH, GRID_SIZE, IMAGE_SIZE, DEVICE, NUM_CLASSES, SINGLE_PLANT_MODE

    IMAGE_PATH = args.image_path
    # IMAGE_PATH = "/scratch/jme3qd/data/plantclef2025/quadrat/images/CBN-can-A6-20230705.jpg" 

    OUTPUT_PATH = args.output_path

    MODEL_TYPE = "lora" 
    MODEL_CHECKPOINT = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
    #PLANT_HOME = "/scratch/jme3qd/data/plantclef2025"

    QUADRAT_DIR = os.path.join(PLANT_HOME, "quadrat/images")
    SPECIES_MAP_FILE = os.path.join(PLANT_HOME, "species_map.csv")
    CLASS_LIST_DIR = os.path.join(PLANT_HOME, "images_max_side_800") 
    FINE_TUNED_MODEL_PATH = os.path.join(PLANT_HOME, "lucas/models/final_lora_classifier_model.pth")

    GRID_SIZE = tuple(args.grid_size)
    IMAGE_SIZE = 518
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 7806 
    SINGLE_PLANT_MODE = args.single_plant_mode 


    class_ids = load_class_names(CLASS_LIST_DIR)
    species_map = load_species_map(SPECIES_MAP_FILE)
    model = load_model(DEVICE, len(class_ids))
    visualize_gradcam(IMAGE_PATH, model, class_ids, species_map, DEVICE)

if __name__ == "__main__":
    main()