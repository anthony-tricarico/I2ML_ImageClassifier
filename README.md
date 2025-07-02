# I2ML_ImageClassifier

This project is focused on **image retrieval and classification** using deep learning models (ResNet, CLIP, etc.) on datasets such as Oxford Pets. It includes code for training, feature extraction, and evaluating retrieval performance using various metrics.

---

## Project Structure

```
I2ML_ImageClassifier/
│
├── accuracies/                # Notebooks and scripts for evaluating model accuracy and retrieval metrics
│   └── oxford_accuracies.ipynb
│
├── CLIP-14_ForwardPass.ipynb  # CLIP model feature extraction and retrieval notebook
├── CLIP-14_ForwardPass 2.ipynb
│
├── FT_CLIP-14.ipynb           # Fine-tuning CLIP notebook
├── FT_CLIP-14 2.ipynb
│
├── models/                    # Model definitions and utilities
│   ├── DinoV2/
│   └── ResNet/
│       ├── resnet 2.ipynb     # ResNet retrieval and training notebook
│       └── utils/
│
├── oxford_pets/               # Oxford Pets dataset (images and splits)
│   └── images/
│
├── data/                      # Additional data, mappings, and splits
│
├── test-2/                    # Alternative test set (gallery/query split)
│
├── train/                     # Training images, organized by class
│
├── requirements.txt           # Python dependencies
├── README.md                  # (You are here)
└── retrieval_results.json     # Example retrieval results (JSON)
```

---

## Main Files & Folders

- **accuracies/**: Contains notebooks for evaluating retrieval and classification accuracy, e.g., `oxford_accuracies.ipynb` (ResNet and CLIP evaluation on Oxford Pets).
- **ArcFace/**: Implementation of the ArcFace loss/model for metric learning.
- **CLIP-14_ForwardPass.ipynb**: Notebook for extracting features and evaluating retrieval with the base CLIP model.
- **FT_CLIP-14.ipynb**: Notebook for fine-tuning CLIP on your dataset.
- **models/ResNet/resnet 2.ipynb**: Notebook for training and evaluating a ResNet-based retrieval model.
- **oxford_pets/**: Contains the Oxford Pets dataset images and (optionally) split folders for gallery/query.
- **data/**: Contains additional data, such as mapping files and split information.
- **test-2/**: Contains a test set with gallery and query folders for retrieval evaluation.
- **train/**: Contains training images, organized by class.
- **retrieval_results.json**: Example output file with retrieval results in JSON format.

---

## How to Use

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the dataset:**

   - Place the Oxford Pets images in `oxford_pets/images/`.
   - Use the provided scripts/notebooks to split images into `gallery` and `query` folders.

3. **Run retrieval experiments:**

   - Use `models/ResNet/resnet 2.ipynb` for ResNet-based retrieval.
   - Use `CLIP-14_ForwardPass.ipynb` for CLIP-based retrieval.
   - Use `accuracies/oxford_accuracies.ipynb` to compare models and compute metrics (Recall@k, Accuracy@1, Precision@k).

4. **Evaluate results:**
   - Retrieval results are saved as JSON files (e.g., `retrieval_results.json`).
   - Metrics are printed in the notebooks.

---

## Example Metrics

- **Recall@k**: Fraction of queries for which at least one correct image is in the top-k retrieved.
- **Accuracy@1**: Fraction of queries where the top-1 retrieved image is correct.
- **Precision@k**: Average fraction of correct images in the top-k retrieved.

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- transformers
- scikit-learn
- numpy
- PIL

(See `requirements.txt` for full details.)

---

## Notes

- The project supports both CPU and GPU (CUDA/MPS) for inference.
- For large models like CLIP-Large, GPU is highly recommended for reasonable speed.
- You can add your own models or datasets by following the structure in the provided notebooks.
