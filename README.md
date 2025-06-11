# Pokémon Image Classification
<img src="./image.png" alt="PokéNet AI" width="400"/>

A comprehensive PyTorch pipeline for recognizing all **151 original Pokémon** from **15K+ images**, blending deep learning architectures and classical ML for accuracy, efficiency, and interpretability.


---

##  Features

- **Dataset Preparation**  
  - 15,100+ Pokémon images, balanced across 151 species  
  - Resized to 240×240, normalized  
  - 70% train / 15% validation / 15% test splits  

- **Model Zoo**  
  - **Custom CNNs**: from simple Conv2D stacks to deep residual variants  
  - **ConvMixer**: hybrid CNN inspired by Vision Transformers  
    - 71.09% test accuracy, ~0.7 M parameters  
  - **ResNet-18 & ViT**  
    - From-scratch vs. pretrained/fine-tuned  
    - Best: ViT frozen-head fine-tuning at 82.71% accuracy  

- **Classical ML on Features**  
  - Extract 512-dim embeddings from ResNet-18 head  
  - Train XGBoost, Random Forest, and ExtraTrees for comparison  

- **Interpretability**  
  - Grad-CAM visualizations to inspect model attention  


---

## 🔧 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/liel2544/Pok-mon-Image-Classification.git
   cd pokemon-image-classification

##  Results Snapshot

| Model                                                         | Test Accuracy | Num of Params / Details                         |
| ------------------------------------------------------------- | ------------: | ------------------------------------------------ |
| **Basic CNN**                                                 |         0.66% | 29.6 M                                           |
| **Custom CNN Modern (Residual CNN)**                          |        11.89% | 129 M                                            |
| **ConvMixer Model**                                           |        71.09% | 700 K                                            |
| **ResNet-18 From Scratch**                                    |        59.24% | 11.3 M                                           |
| **Vision Transformer (ViT) From Scratch**                     |         8.44% | 85.9 M                                           |
| **ResNet-18 Fine-Tuned (experiment 1)**                       |        69.58% | 11.3 M                                           |
| **Pretrained ViT Staged Fine-Tune (experiment 1)**            |         3.85% | 85.9 M                                           |
| **XGBoost on ResNet Features (experiment 1)**                 |        42.00% | n_estimators=200, max_depth=16                   |
| **Random Forest on ResNet Features (experiment 1)**           |        43.00% | n_estimators=300, max_depth=24                   |
| **Extra Trees on ResNet Features (experiment 1)**             |        46.00% | n_estimators=300, max_depth=24                   |
| **ResNet-18 Fine-Tuned with Frozen Deep Head (experiment 2)** |        59.37% | 11.6 M                                           |
| **Pretrained ViT Staged Fine-Tune (experiment 2)**            |        82.71% | 86.4 M                                           |
| **XGBoost on Fine-Tuned ResNet Head Features (experiment 2)**|        58.00% | n_estimators=300, max_depth=24                   |
| **Random Forest on Fine-Tuned ResNet Head Features (experiment 2)** |    63.00% | n_estimators=300, max_depth=24                   |
| **Extra Trees on Fine-Tuned ResNet Head Features (experiment 2)**   |    66.00% | n_estimators=300, max_depth=24                   |

