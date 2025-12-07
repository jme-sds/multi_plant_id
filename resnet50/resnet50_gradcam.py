import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm
import numpy as np
import cv2
import csv
import random
import glob
import matplotlib.pyplot as plt

# ===============================================================================
# Define the Grad-CAM Class for ResNet50 (with Multi-label classification head)
# ===============================================================================

class GradCAM:
    '''
    This class will handle the forward pass, 
    registering hooks for gradients and activations, 
    and generating the Grad-CAM heatmap.
    '''    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = None
        self.gradients = None
        self.activations = None

        # Register hooks
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                module.register_forward_hook(self._save_activation)
                module.register_backward_hook(self._save_gradient)
                self.target_layer = module
                break
        if self.target_layer is None:
            raise ValueError(f"Target layer '{target_layer_name}' not found in model.")

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple (grad_wrt_output,)
        self.gradients = grad_output[0]

    def generate_heatmaps(self, input_image, device, threshold=0.1):
        """
        Generates Grad-CAM heatmaps for specified target classes.
        If target_classes is None, it generates heatmaps for the top_k predicted classes.
        Returns a 'all_heatmaps_data' list of (class_index, class_probability, heatmap) tuples.
        """
        all_heatmaps_data = []

        self.model.eval() # Ensure model is in evaluation mode

        # Load and preprocess image
        preprocess = get_transforms()
        input_tensor = preprocess(input_image).unsqueeze(0).to(device)
        input_tensor.requires_grad = True # Ensure gradients are tracked for input
        
        # Perform a single forward pass to get predictions for a single image input
        initial_logits = self.model(input_tensor)
        initial_probabilities = torch.sigmoid(initial_logits)  # convert to probabilities in the multilabel prediction way
        preds_multihot = (initial_probabilities > threshold).float()  # threshold
        preds_idx = np.where(preds_multihot[0] == 1)[0].tolist()   # get the indices based on multihot

        if len(preds_idx) == 0:
            print("No target classes found.")
            return [], []
        else:
            # target_classes_to_process = [(idx, initial_probabilities[idx].item()) for idx in preds_idx]
            # target_classes_to_process = [(idx, initial_probabilities[0, idx].item()) for idx in preds_idx]
            target_classes_to_process = [(idx, initial_probabilities[0][idx].item()) for idx in preds_idx]
        
        for class_idx, class_prob in target_classes_to_process:
            self.model.zero_grad() # Clear gradients before each backward pass

            # Re-run forward pass to ensure a fresh computation graph for each backward call
            output_for_backward = self.model(input_tensor)

            # Create one-hot vector for the specific class
            one_hot = torch.zeros_like(output_for_backward)
            one_hot[0][class_idx] = 1

            # Perform backward pass
            output_for_backward.backward(gradient=one_hot, retain_graph=True) # Retain graph as we are doing multiple backward passes

            # Get feature map and gradients
            gradients = self.gradients.cpu().data.numpy()[0]
            activations = self.activations.cpu().data.numpy()[0]

            # Global average pooling of gradients
            weights = np.mean(gradients, axis=(1, 2)) # (C,)

            # Weighted combination of feature maps
            cam = np.zeros(activations.shape[1:], dtype=np.float32) # (H, W)
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]

            # ReLU for only positive contributions
            cam = np.maximum(cam, 0)
            if cam.max() == 0: # Handle cases where max is 0 to avoid division by zero
                cam = np.zeros_like(cam)
            else:
                cam = cam / cam.max() # Normalize to 0-1

            all_heatmaps_data.append((class_idx, class_prob, cam))

        return all_heatmaps_data, preds_idx

# ==========================================
# UTILITIES
# ==========================================

def get_class_names(preds_idx, idx_to_cls_mapdict):
    preds_names = []
    for idx in preds_idx:
            cls_name = idx_to_cls_mapdict[str(idx)]
            preds_names.append(cls_name)            
    return preds_names

def load_model(model, device):
    print("Loading ResNet50 model...")

    # CRITICAL FIX FOR GRADCAM: Unfreeze parameters
    # Even though we are in eval mode, we need PyTorch to track gradients
    # through the backbone to the input image for GradCAM to work.
    for param in model.parameters():
        param.requires_grad = True

    model.to(device)
    model.eval() # Set model to evaluation mode
    return model

def get_tiles(img_path, tiles_per_side=3): 
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    tile_w = w / tiles_per_side
    tile_h = h / tiles_per_side

    tiles = []

    for row in range(tiles_per_side):
        for col in range(tiles_per_side):
            left = int(col * tile_w)
            top = int(row * tile_h)
            right = int((col + 1) * tile_w)
            bottom = int((row + 1) * tile_h)
            tile = img.crop((left, top, right, bottom))            
            tiles.append(tile)
    return tiles

def get_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize(size=(image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ==========================================
# VISUALIZATION
# ==========================================

def visualize_gradcam(image, model, idx_to_cls_mapdict, device, threshold=0.1, target_layer_name='layer4'):  # Common target layer for ResNet50
    '''
    "image" is each tile of a quadrat from get_tiles().
    '''
    grad_cam = GradCAM(model, target_layer_name)
    all_heatmaps_data, preds_idx = grad_cam.generate_heatmaps(image, device, threshold=threshold)
    
    class_names = get_class_names(preds_idx, idx_to_cls_mapdict)

    # Convert raw_image to numpy array for blending
    img_np = np.array(image)

    plt.figure(figsize=(8, 5 * (len(all_heatmaps_data) + 1))) # +1 for the merged heatmap
    # --- Display each generated Grad-CAM heatmap ---
    # for i, (class_idx, class_prob, heatmap) in enumerate(all_heatmaps_data):
    #     # Resize heatmap to original image size
    #     heatmap_resized = cv2.resize(heatmap, (image.width, image.height))
    #     heatmap_colored = np.uint8(255 * heatmap_resized)
    #     heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    #     # Blend heatmap with original image
    #     superimposed_img = heatmap_colored * 0.4 + img_np
    #     superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))

    #     plt.subplot(len(all_heatmaps_data) + 1, 1, i + 1)
    #     plt.imshow(superimposed_img)
    #     plt.title(f"Grad-CAM Heatmap for {class_names[class_idx]} (Prob: {class_prob:.2f})")
    #     plt.axis('off')

    # --- Merge heatmaps for predicted classes ---
    merged_classes = class_names

    if merged_classes:
        merged_heatmap = np.zeros_like(all_heatmaps_data[0][2]) # Initialize with shape of first heatmap   (class_idx, class_prob, cam)
        for class_idx, class_prob, heatmap in all_heatmaps_data:
            merged_heatmap += heatmap
            
        # Normalize the merged heatmap      
        if merged_heatmap.max() > 0:
            merged_heatmap = merged_heatmap / merged_heatmap.max()

        # Resize and colorize the merged heatmap
        merged_heatmap_resized = cv2.resize(merged_heatmap, (image.width, image.height))
        merged_heatmap_colored = np.uint8(255 * merged_heatmap_resized)
        merged_heatmap_colored = cv2.applyColorMap(merged_heatmap_colored, cv2.COLORMAP_JET)

        # Blend merged heatmap with original image
        merged_superimposed_img = merged_heatmap_colored * 0.4 + img_np
        merged_superimposed_img = np.uint8(255 * merged_superimposed_img / np.max(merged_superimposed_img))
        
    else:
        print(f"No classes found with threshold >= {threshold}.")
        merged_superimposed_img = img_np * 0.4 + img_np
        merged_superimposed_img = np.uint8(255 * merged_superimposed_img / np.max(merged_superimposed_img))
   
    plt.subplot(len(all_heatmaps_data) + 1, 1, len(all_heatmaps_data) + 1)
    plt.imshow(merged_superimposed_img)
    plt.title(f"Merged Grad-CAM Heatmap for {', '.join(merged_classes)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# def main(model, img_path, idx_to_cls_mapdict, threshold=0.2, target_layer_name='layer4'):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     model = load_model(model, device)
    
#     tiles = get_tiles(img_path, tiles_per_side=3, target_size=224)

#     # Generate and visualize GradCAM for each tile
#     for tile in tiles:
#         visualize_gradcam(tile, model, idx_to_cls_mapdict, device, threshold=threshold, target_layer_name=target_layer_name)  # Common target layer for ResNet50


# if __name__ == "__main__":
#     main()

