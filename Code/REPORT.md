# Benchmark Report — Operation Cyber-Histology

## Overview

This report presents the final test-set performance of three convolutional architectures (AlexNet, VGG16, ResNet18) across four MedMNIST-derived diagnostic image datasets (cells, chest, lesions, orgs), following full pipeline reconstruction and bug remediation. All models were trained for 10 epochs using the Adam optimizer (learning rate 0.001), a dropout rate of 0.5, ReLU activations, and batch size 64. Validation-best checkpoints (selected by highest validation accuracy) were used for test-set evaluation, not final-epoch weights.

## Benchmark Results

| Dataset | Model | Test Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) | Target | Status |
|---|---|---|---|---|---|---|---|
| cells | AlexNet | 92.25% | 92.20% | 91.39% | 91.56% | 90% | PASS |
| cells | VGG16 | 93.10% | 92.01% | 93.23% | 92.40% | 90% | PASS |
| cells | **ResNet18** | **94.62%** | 93.86% | 94.33% | 93.80% | 90% | PASS |
| chest | AlexNet | 88.62% | 91.44% | 85.17% | 87.02% | 87% | PASS |
| chest | **VGG16** | **89.74%** | 92.74% | 86.41% | 88.32% | 87% | PASS |
| chest | ResNet18 | 87.02% | 90.90% | 82.86% | 84.92% | 87% | PASS |
| lesions | **AlexNet** | **73.87%** | 51.02% | 36.94% | 40.11% | 67% | PASS |
| lesions | VGG16 | 70.17% | 28.69% | 29.69% | 27.60% | 67% | PASS |
| lesions | ResNet18 | 72.67% | 47.14% | 43.29% | 40.27% | 67% | PASS |
| orgs | AlexNet | 87.18% | 86.10% | 84.89% | 85.11% | 83% | PASS |
| orgs | VGG16 | 89.07% | 87.94% | 86.95% | 87.08% | 83% | PASS |
| orgs | **ResNet18** | **90.59%** | 89.69% | 89.01% | 89.08% | 83% | PASS |

**Bold** entries indicate the best-performing model for each dataset. All 12 dataset/model combinations exceeded their minimum accuracy targets.

## Analysis

**Accuracy vs. macro F1-score divergence on `lesions`:** all three models on the `lesions` dataset show a substantial gap between test accuracy (70-74%) and macro F1-score (27-40%), despite comfortably passing the 67% accuracy target. This pattern is a strong indicator of class imbalance — the dataset likely contains a small number of dominant lesion categories and several rare ones. Since accuracy weighs every prediction equally, a model can score well by reliably identifying common classes while performing poorly on rare ones. Macro-averaged precision, recall, and F1 weight every class equally regardless of frequency, exposing this weakness. Accuracy alone would have overstated model quality on this dataset.

**ResNet18 performs best on high-channel-complexity or high-class-count datasets:** ResNet18 wins on `cells` (8 classes, RGB) and `orgs` (11 classes, grayscale) — the two datasets with the most output classes. Its residual connections likely help it optimize more effectively as classification complexity increases, compared to the plain convolutional stacks in AlexNet and VGG16.

**VGG16 wins on `chest`, a simpler binary classification task:** with only 2 classes, `chest` may not benefit as strongly from ResNet18's depth and residual connections; VGG16's deeper feature extraction without residual shortcuts was sufficient here, and outperformed ResNet18 by ~2.7 percentage points in accuracy and ~3.4 in F1.

**AlexNet performs surprisingly well on `lesions`:** despite being the simplest architecture, AlexNet achieved the highest accuracy and F1-score on this dataset. This may indicate that lesion classification benefits from simpler decision boundaries at this scale, or that the deeper models (VGG16, ResNet18) are more prone to overfitting on this dataset within only 10 epochs, given lesions' apparent class imbalance and comparatively harder classification task.

## Architectural Recommendations

| Dataset | Recommended Model | Rationale |
|---|---|---|
| cells | ResNet18 | Highest accuracy and F1; benefits from residual learning on 8-class RGB classification |
| chest | VGG16 | Highest accuracy and F1 on this binary task; ResNet18's added complexity did not translate to better performance here |
| lesions | AlexNet | Highest accuracy and F1, though all models show weak macro-F1 due to likely class imbalance — further data balancing (e.g. class-weighted loss, oversampling) is recommended regardless of model choice |
| orgs | ResNet18 | Highest accuracy and F1 on the highest-class-count dataset (11 classes) |

**General recommendation:** for datasets with a higher number of classes or more complex diagnostic patterns (`cells`, `orgs`), ResNet18's residual connections provide a consistent advantage. For simpler binary tasks (`chest`), a non-residual deep architecture (VGG16) was sufficient and slightly outperformed ResNet18. For `lesions`, model choice appears secondary to addressing class imbalance — this should be prioritized in future iterations regardless of which architecture is ultimately deployed.

## Limitations and Future Work

- Results reflect only 10 training epochs per model; extending training (e.g. to 20 epochs, matching the original hyperparameter registry) may close some of the observed AlexNet/VGG16/ResNet18 performance gaps, particularly on `lesions`.
- Macro-averaged F1-score results for `lesions` suggest that per-class performance analysis (confusion matrices, per-class F1) would provide more actionable guidance than aggregate metrics alone.
- No learning rate scheduling or data augmentation was applied in this benchmark; both are common techniques that could improve generalization, particularly for the underperforming `lesions` dataset.