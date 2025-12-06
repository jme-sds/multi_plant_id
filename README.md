# multi_plant_id
Plant species identification using deep learning methods to address the PlantCLEF 2025 Kaggle challenge.

# Abstract

This project aims at the PlantCLEF2025 Kaggle/LifeCLEF challenge on multi-species plant identification in unlabeled vegetation quadrat images. We describe a pipeline for zero-shot multi-species plant identification using a fine-tuned pretrained Vision Transformer (ViT) model combined with methods of high-resolution image tiling, false positive reduction, and prediction aggregation.


### Ablations

We did a lot of study on a baseline resnet50 model studying various ablations in order to improve performance, the full implementation can be seen in the `resnet50` file directory.

The code for the ablation studies on the ViT model can be seen in the `main` folder and can be run using the slurm script that is included.

All of the data can be loaded and created as datasets using the methods found in the `loading` folder. There are two seperate files for loading, one for single plants and one for quadrats

The `dino` and `knn` folders are used in the creating of our final ViT model. The final model leverages a trained classification head as well as a FAISS index with knn voting schema.

## Final Model 
The final model leverages a trained classification head as well as 3x4 tiled grid pattern and knn voting.

We provide our best fine-tuned model (DINOv2 + LoRA) via GitHub Releases.

| Model | Backbone | Validation Score | Download |
| :--- | :--- | :--- | :--- |
| **PlantCLEF-2025-Best** | DINOv2 (ViT-B/14) | **0.28 F1** | [Download Checkpoint]([./releases](https://github.com/jme-sds/multi_plant_id/releases/tag/v1.0.0)) |

###  How to Load

1. **Download the weights** from the [Releases Page]().
2. **Install dependencies:** `pip install torch timm peft`
3. **Run this Python script:**

```python
import torch
import timm
from peft import PeftModel

def load_plantclef_model(checkpoint_path="final_fine_tuned_model.pth", device="cuda"):
    print(f"Loading model from {checkpoint_path}...")
    
    # 1. Initialize Base DINOv2 Model
    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m", 
        pretrained=False, 
        num_classes=7806  # PlantCLEF class count
    )
    
    # 2. Load the State Dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle standard vs. wrapped state dicts
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 3. Load weights into model
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Load status: {msg}")
    
    model.to(device)
    model.eval()
    return model

# Usage
# model = load_plantclef_model("path/to/downloaded/final_fine_tuned_model.pth")
```

### Environment Setup
The scripts rely on a central environment variable `PLANT_HOME` to locate datasets and save models. You must set this before running any scripts.
```
export PLANT_HOME="/path/to/your/dataset/root"
```
#### Directory Structure

Ensure your data is organized as follows within `PLANT_HOME`:
```
PLANT_HOME/
├── images_max_side_800/       # Single-plant training images (folders by species_id)
├── quadrat/images/            # Test quadrat images (.jpg)
├── lucas/images/              # Unlabeled LUCAS pseudo-quadrats
├── lucas/models/              # Output directory for trained models
├── dinov2_model/              # Challenge-provided weights
│   └── model_best.pth.tar
└── species_mapping.csv        # (Optional) Map of species_id -> species_name
```

### Training
#### Train Baseline (Linear Probe)

This script trains a simple classification head on top of the frozen DINOv2 backbone. It is fast and memory-efficient, serving as a robust baseline.

```
python train_classifier.py
```
- Output: Saves baseline_fine_tuned.pth to the current directory.

#### Train LoRA (SSL + Supervised)

This runs the complete domain adaptation pipeline:

Stage 1 (SSL): SimCLR contrastive learning on unlabeled LUCAS data (default 50 epochs).

Stage 2 (SFT): Supervised classification on labeled single plants (default 5 epochs).
```
python lora.py \
  --ssl_epochs 50 \
  --sft_epochs 5 \
  --lora_path "lora_lucas_ssl_weights" \
  --batch_size_per_gpu 32
```
- Output: Saves the full model (adapters + head) to `PLANT_HOME/lucas/models/final_fine_tuned_model.pth`.

## Inference & Submission

These scripts generate Kaggle-formatted submission files (.csv) by tiling high-resolution quadrats and pooling predictions.

### Baseline Inference (Multi-Grid Pooling)

Runs inference using the Baseline model across multiple grid sizes (e.g., 2x2 and 3x3) and pools the predictions via Union Voting.
```
python baseline_inference_dino.py \
  --grids 2 2 3 3 \
  --fine_tuned_path "baseline_fine_tuned.pth" \
  --output "submission_baseline.csv"
```
### LoRA Inference

Runs inference using the fine-tuned LoRA model on a specific grid size (e.g., 3x4).
```
python lora_inference.py \
  --grid_size 3 4 \
  --model_path "lucas/models/final_fine_tuned_model.pth"
```
## Visualization: Grad-CAM Heatmaps

Visualize which parts of the image the model is focusing on. This script supports generating heatmaps for both the Baseline and LoRA models to compare their attention mechanisms.

### Compare both Baseline and LoRA on a random test image
```
python gradcam.py --grid_size 3 3
```

### Visualize a specific image with the LoRA model
```
python gradcam.py \
  --image_path "/path/to/specific/quadrat.jpg" \
  --output_path "./gradcam_quadrat.jpg"
```