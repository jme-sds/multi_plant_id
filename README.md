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
   ```bash
   wget [https://github.com/jme-sds/multi_plant_id/releases/download/v1.0.0/baseline_fine_tuned.pth](https://github.com/jme-sds/multi_plant_id/releases/download/v1.0.0/baseline_fine_tuned.pth)
   ```
3. **Install dependencies:**
   ```bash
   pip install torch tqdm numpy scipy scikit-learn torchvision torchaudio matplotlib seaborn
   ```
4. **Run this Python script:**

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

Or if you prefer, you can clone this repo!

```bash
git clone [https://github.com/jme-sds/multi_plant_id](https://github.com/jme-sds/multi_plant_id)
```

And install the dependecies.

```bash
pip install -r requirements.txt
```


