# Explainable Deep Learning for Multi-Class Brain Tumour Classification Using MRI Scans

A comparative study of **ResNet50** and **EfficientNetB0** for brain tumour classification with dual explainability methods (**Grad-CAM** and **SHAP**), conducted as a BSc (Hons) Artificial Intelligence dissertation at Northumbria University.

## Overview

This project investigates explainable deep learning for multi-class brain tumour classification from 2D MRI slices. Two pre-trained CNN architectures are compared across two publicly available datasets under controlled experimental conditions. The study integrates Grad-CAM and SHAP to provide both visual and quantitative model interpretability, addressing the transparency requirements for clinical decision support.

### Key Contribution

The primary original contribution is the **empirical quantification of data leakage** caused by random image-level splitting on multi-slice MRI datasets. By comparing random splitting against patient-level splitting, we demonstrate that random splitting inflates accuracy by an average of **7.56 percentage points**, with meningioma F1 inflated by up to **13.44 percentage points**. This finding highlights a widespread methodological concern in medical imaging research where patient-level splitting is not enforced.

## Datasets

| Dataset | Source | Classes | Images | Patients | Split Strategy |
|---------|--------|---------|--------|----------|----------------|
| **Kaggle Brain Tumour MRI Dataset V2** | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | 4 (Glioma, Meningioma, Pituitary, No Tumour) | 6,878 (after MD5 deduplication) | N/A | Random 80/20 |
| **Figshare Brain Tumor Dataset V5** | [Figshare](https://ndownloader.figshare.com/articles/1512427/versions/5) | 3 (Glioma, Meningioma, Pituitary) | 3,064 | 233 | Patient-level 80/20 |

> **Note on Kaggle V2:** Despite claiming deduplication, residual duplicates were discovered and removed via MD5 hash-based cleaning a secondary methodological contribution of this work.

## Results Summary

### Patient-Level Splitting (Figshare)

| Model | Accuracy | Cohen's Kappa | Parameters |
|-------|----------|---------------|------------|
| ResNet50 | 88.10% | 0.8191 | 24,033,347 |
| EfficientNetB0 | 92.06% | 0.8787 | 4,336,255 |

EfficientNetB0 outperforms ResNet50 by **+3.96pp** in accuracy while using **82% fewer parameters**, demonstrating a favourable efficiency–accuracy trade-off for resource-constrained clinical settings.

### Data Leakage Impact (Figshare: Random vs Patient-Level Splitting)

| Model | Random Split Accuracy | Patient-Level Accuracy | Inflation |
|-------|----------------------|----------------------|-----------|
| ResNet50 | 96.09% | 88.10% | +7.99pp |
| EfficientNetB0 | 99.18% | 92.06% | +7.12pp |
| **Average** | — | — | **+7.56pp** |

These results demonstrate that studies reporting 97–99% accuracy on similar datasets without patient-level splitting may be substantially overestimating real-world generalisation.

## Explainability

Two complementary XAI methods are applied to provide multi-perspective model interpretability:

- **Grad-CAM** — Visual heatmaps highlighting tumour-relevant regions in the MRI scan. Applied to `layer4[-1]` (ResNet50) and `conv_head` (EfficientNetB0).
- **SHAP (PartitionExplainer)** — Quantitative pixel-level feature attribution using `inpaint_telea` masker with `max_evals=50`. Provides directional analysis of which image regions contribute positively or negatively to predictions.

## Project Structure

```
brain-tumour-classification-xai/
├── README.md
├── notebooks/
│   ├── Brain_Tumour_Classification_Kaggle.ipynb
│   ├── Brain_Tumour_Classification_Figshare.ipynb
│   └── Brain_Tumour_Cross_Dataset_Comparison.ipynb
├── results/                  # Saved JSON result files
└── figures/                  # Grad-CAM and SHAP visualisations
```

## Environment & Reproducibility

**Platform:** Google Colab (NVIDIA Tesla T4 GPU, 15 GB VRAM)

**Key dependencies:**
- Python 3.10
- PyTorch 2.5.1
- timm (1.0+) — pre-trained model access
- pytorch-grad-cam — Grad-CAM implementation
- shap — SHAP PartitionExplainer
- scikit-learn — evaluation metrics
- h5py, scipy — Figshare dataset processing

**Reproducibility:** Random seeds set to `42` for PyTorch and NumPy across all experiments.

**Preprocessing:** Both models use ImageNet normalisation values (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) with images resized to 224×224.

## Training Configuration

- **Optimiser:** Adam (ResNet50 lr=1e-4, EfficientNetB0 lr=3e-4)
- **Scheduler:** ReduceLROnPlateau monitoring test loss (patience=3)
- **Early stopping:** Patience=5 on validation loss
- **Fine-tuning:** Full fine-tuning of all layers (no gradual unfreezing)
- **Batch size:** 32
- **Dropout:** 0.5

## Citation

If you use this work, please cite:

```
Alshammari, H. (2026). Explainable Deep Learning for Multi-Class Brain Tumour
Classification Using MRI Scans. BSc (Hons) Dissertation, Northumbria University.
Supervised by Dr. Anas Althobaiti.
```

## Licence

This project is submitted as academic coursework. Please contact the author for reuse permissions.

## Acknowledgements

This work was supervised by Dr. Anas Althobaiti at Northumbria University. The Kaggle Brain Tumour MRI Dataset V2 and Figshare Brain Tumor Dataset V5 are gratefully acknowledged.
