Dogs vs. Cats Classification — VGG-Style CNN
A convolutional neural network built from scratch with TensorFlow/Keras, with configurable image preprocessing, data augmentation, and reproducible training experiments.
This project implements a VGG-inspired CNN for binary image classification on the Dogs vs. Cats dataset. Beyond training a baseline classifier, the notebook provides a configurable pipeline to investigate the impact of image filtering, augmentation, regularization, and architectural choices on validation performance.
Results
Metric	Recorded result
Best-run validation accuracy	87.95%
Validation AUC	0.9548
Model parameters	1,175,393
Best epoch (by validation loss)	40
Epochs run	48 / 50 (early stopping)
Training time	41.1 minutes
Training / validation subset	5,000 / 2,000 images
Input resolution	224 × 224 RGB
These results are from the `vgg_baseline_n2500_lr5e-4_ls0.1` run recorded in the notebook (seed 41), not an average across random seeds. Accuracy is measured on the selected validation set, not the unlabeled test set.
Model architecture
The CNN is implemented using Keras layers rather than a pretrained VGG model.
```text
RGB image (224 × 224 × 3)
         │
         ▼
Conv block 1: 2 × [3×3 Conv(32)  → BatchNorm → ReLU] → MaxPool
         ▼
Conv block 2: 2 × [3×3 Conv(64)  → BatchNorm → ReLU] → MaxPool
         ▼
Conv block 3: 2 × [3×3 Conv(128) → BatchNorm → ReLU] → MaxPool
         ▼
Conv block 4: 2 × [3×3 Conv(256) → BatchNorm → ReLU] → MaxPool
         ▼
Global Average Pooling → Dropout(0.4)
         ▼
Dense(1, sigmoid) → P(dog)
```
Four convolutional blocks with increasing channel counts: 32, 64, 128, 256
Two 3×3 convolutions per block, with batch normalization and ReLU
Global average pooling by default; a `Flatten` + dense-head alternative is available for comparison
Sigmoid binary classifier, with class labels `0 = cat` and `1 = dog`
He-normal initialization and float32 classification output, including under mixed-precision training
Data pipeline and experiments
The `tf.data` input pipeline handles image decoding, RGB resizing, shuffling, batching, caching, and prefetching. Training inputs are randomly augmented; validation and test inputs remain unaugmented.
Available augmentations
Horizontal flipping, rotation, zoom, and translation
Optional simulated low-resolution downscaling
Optional per-image 3×3 median filtering or Laplacian sharpening
Training configuration
AdamW optimizer; initial learning rate `5e-4`, weight decay `1e-4`
Binary cross-entropy with label smoothing `0.1`
Batch size `32`, maximum `50` epochs
Early stopping on validation loss (patience `8`)
ReduceLROnPlateau (patience `3`)
Best-model checkpointing, experiment logs, and learning curves
Optional GPU mixed precision and a three-seed experiment sweep
The notebook supports controlled comparisons of augmentation/filter settings and the global-average-pooling versus flattening classification head. Only results actually saved in the notebook are reported above; optional configurations are not claimed as completed experiments.
Running the notebook
Requirements
Google Colab (GPU runtime recommended)
Python with TensorFlow/Keras, NumPy, pandas, SciPy, and Matplotlib
Google Drive access for the dataset and saved outputs
The recorded environment used TensorFlow 2.21.0 and Keras 3.13.2.
Dataset layout
Upload `IE4483.zip` to `MyDrive/IE4483/`. The zip must extract into this structure:
```text
datasets/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
    └── <numeric_image_id>.jpg
```
The notebook checks for these available images: 10,000 training images per class, 2,500 validation images per class, and 500 unlabeled test images. By default, it samples 2,500 training and 1,000 validation images per class using seeded selection.
Steps
Open `IE4483_CNN.ipynb` in Google Colab.
Select Runtime → Change runtime type → GPU.
Place `IE4483.zip` in `MyDrive/IE4483/`.
Change configuration values in Section 1 if needed, then select Runtime → Run all.
Review the validation metrics, saved training curves, model checkpoint, and `results.csv` in your Drive project folder.
The notebook defines `write_submission(model)` to generate a test-prediction CSV (`id,label`, sorted by numeric image ID). To generate it after training, explicitly run:
```python
write_submission(baseline_model)
```
This writes `submission.csv` to `MyDrive/IE4483/`; generating it is not currently called automatically in the notebook.
Outputs
```text
MyDrive/IE4483/
├── results.csv             # Experiment configurations and validation metrics
├── checkpoints/            # Best saved .keras checkpoints
├── figures/                # Architecture diagram and training curves
└── submission.csv          # Generated when write_submission(...) is called
```
What this project demonstrates
Implementing and training a CNN architecture from first principles using Keras layers
Engineering an efficient, configurable image input pipeline with `tf.data`
Comparing preprocessing and augmentation strategies under controlled settings
Tracking model generalization using validation accuracy, AUC, and loss
Structuring experiments with repeatable seeds, automatic checkpoints, and performance logs
Scope
This repository documents the CNN baseline and its configurable training/validation pipeline. It does not currently include Vision Transformer or pretrained ResNet experiments, embedded deployment, or on-device inference benchmarks.
