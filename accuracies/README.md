# Accuracies Directory

This directory contains evaluation results and analysis of pre-trained models' performance on the **Oxford Pet Dataset** prior to any fine-tuning.

## Purpose

The files in this directory showcase the baseline performance of various pre-trained computer vision models when applied to the Oxford Pet dataset without any domain-specific fine-tuning. This serves as a benchmark to understand how well these models perform "out-of-the-box" on the pet classification task.

## Contents

### `oxford_accuracies.ipynb`

This Jupyter notebook contains comprehensive evaluations of three popular pre-trained models:

1. **ResNet-18** - A classic convolutional neural network architecture
2. **CLIP (ViT-L/14@336px)** - A vision-language model from OpenAI
3. **EfficientNet-B0** - An efficient CNN architecture optimized for mobile/edge devices

## Evaluation Methodology

The notebook implements an image retrieval task where:

- **Query Set**: One image per breed is randomly selected as a query
- **Gallery Set**: All remaining images serve as the searchable gallery
- **Task**: For each query image, find the most similar images in the gallery
- **Evaluation**: Measure how well the model retrieves images of the same breed

## Performance Metrics

Each model is evaluated using three key metrics:

- **Recall@10**: Percentage of queries where at least one correct match appears in the top-10 results
- **Accuracy@1**: Percentage of queries where the top-1 result is correct
- **Precision@10**: Average fraction of correct matches in the top-10 results

## Results Summary

| Model               | Recall@10 | Accuracy@1 | Precision@10 |
| ------------------- | --------- | ---------- | ------------ |
| **CLIP**            | 1.0000    | 0.9730     | 0.9027       |
| **ResNet-18**       | 1.0000    | 0.9189     | 0.8568       |
| **EfficientNet-B0** | 0.9730    | 0.8919     | 0.8784       |

## Key Findings

- **CLIP** demonstrates the best overall performance, achieving near-perfect recall and the highest accuracy
- **ResNet-18** shows solid performance with perfect recall but lower precision
- **EfficientNet-B0** has the lowest performance among the three models tested

## Usage

To reproduce these results or run your own evaluations:

1. Ensure you have the Oxford Pet dataset properly organized
2. Install required dependencies (PyTorch, torchvision, transformers, scikit-learn)
3. Run the notebook cells sequentially
4. Modify the evaluation parameters as needed for your specific use case

## Dataset Structure

The evaluation expects the Oxford Pet dataset to be organized as follows:

```
oxford_pets/
├── images/
│   ├── gallery/          # Gallery images organized by breed
│   │   ├── breed1/
│   │   ├── breed2/
│   │   └── ...
│   └── query/            # Query images (one per breed)
```

This baseline evaluation provides crucial insights for understanding model performance before fine-tuning and helps inform decisions about which models to prioritize for domain-specific adaptation.
