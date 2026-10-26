# Plant disease classifier: Image Classification with a CNN

A convolutional neural network (CNN) built with TensorFlow/Keras that classifies
224×224 RGB images into **38 classes**. The model was trained in Google Colab
on a dataset downloaded from Kaggle.

## Results

| Metric               | Value  |
|----------------------|--------|
| Validation accuracy  | 89.17% |
| Validation loss      | 0.3399 |
| Training accuracy    | ~95.6% (at the best epoch) |
| Input size           | 224 × 224 × 3 |
| Number of classes    | 38 |

Training stopped automatically at epoch 5 using early stopping. The final
model uses the weights from epoch 2, which had the lowest validation loss.

## Dataset

- **Source:** [[Kaggle dataset name and link]](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Classes:** 38
- **Training images:** ~43,400 (1358 batches of 32)
- **Validation images:** ~10,800 (339 batches of 32)
- **Preprocessing:** images resized to 224×224 and rescaled to [0, 1]

## Model Architecture

| Layer                 | Output Shape      | Parameters |
|-----------------------|-------------------|-----------:|
| Conv2D (32, 3×3, ReLU)| (222, 222, 32)    | 896 |
| MaxPooling2D (2×2)    | (111, 111, 32)    | 0 |
| Conv2D (64, 3×3, ReLU)| (109, 109, 64)    | 18,496 |
| MaxPooling2D (2×2)    | (54, 54, 64)      | 0 |
| Flatten               | 186,624           | 0 |
| Dense (256, ReLU)     | 256               | 47,776,000 |
| Dense (64, ReLU)      | 64                | 16,448 |
| Dense (38, Softmax)   | 38                | 2,470 |

**Total parameters:** ~47.8 million

## Training Setup

- **Optimizer:** Adam (learning rate: (0.0005) 
- **Loss:** Categorical cross-entropy
- **Batch size:** 32
- **Max epochs:** 100
- **Callbacks:** EarlyStopping (`monitor='val_loss'`, `patience=3`, `restore_best_weights=True`)

## Training Progress

| Epoch | Train Acc | Val Acc | Val Loss |
|------:|----------:|--------:|---------:|
| 1 | 0.7264 | 0.8516 | 0.4834 |
| 2 | 0.9077 | 0.8917 | **0.3398** |
| 3 | 0.9564 | 0.8905 | 0.3614 |
| 4 | 0.9734 | 0.8946 | 0.3981 |
| 5 | 0.9809 | 0.8955 | 0.3981 |

## Observations

- Training accuracy reaches ~98% while validation accuracy plateaus near 89%,
  which indicates **overfitting**. Most parameters sit in the first Dense layer
  (~47.7M of 47.8M).
- Early stopping prevented further overfitting by restoring the best epoch.

## Possible Improvements

- Add data augmentation (rotation, zoom, flips) to the training generator
- Add Dropout layers after the Dense layers
- Replace `Flatten()` with `GlobalAveragePooling2D()` to cut parameters sharply
- Add more Conv + MaxPooling blocks and BatchNormalization
- Use `ReduceLROnPlateau` to lower the learning rate when progress stalls
- Try transfer learning (MobileNetV2, EfficientNet, ResNet50)

## Requirements

- Python 3.10+
- TensorFlow / Keras
- NumPy, Matplotlib
- kaggle (to download the dataset)

```bash
pip install tensorflow numpy matplotlib kaggle
```

## How to Run

1. Open the notebook in Google Colab (GPU runtime recommended:
   Runtime → Change runtime type → T4 GPU).
2. Add your Kaggle API token (as a Colab Secret) and download the dataset.
3. Run the cells in order: data loading, model definition, training, evaluation.

## Saving and Loading the Model

```python
model.save("model.h5")

from tensorflow.keras.models import load_model
model = load_model("model.h5")
```

## Project Structure

```
├── notebook.ipynb    # training and evaluation code
├── model.keras       # trained model (optional)
└── README.md
```

## Author

Deepak
