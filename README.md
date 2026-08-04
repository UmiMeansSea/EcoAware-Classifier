# EcoAware-Classifier 🌱

A lightweight PyTorch image classifier (CIFAR-10) designed for energy efficiency, carbon footprint tracking, and frugal AI principles. 

`EcoAware-Classifier` measures environmental impact (CO₂ emissions) during model training using [CodeCarbon](https://codecarbon.io/) and establishes a baseline CNN architecture for green, low-compute Deep Learning research.

---

## 🚀 Features

- **Convolutional Neural Network (SimpleCNN)**: Built using PyTorch for classifying 10 object categories in CIFAR-10.
- **Emissions Tracking**: Integrated with CodeCarbon to measure exact electricity consumption and carbon footprint during training cycles.
- **Frugal AI Focus**: Micro-batch processing (`batch_size=64`) to prevent high GPU/CPU memory consumption.
- **Model Checkpointing**: Saves full precision baseline weights (`heavy_model.pth`) for further quantization and model compression experiments.

---

## 🛠️ Project Structure

```text
EcoAware-Classifier/
│
├── eco_classifier.py    # Main training loop with PyTorch CNN & CodeCarbon tracking
├── .gitignore           # Ignores downloaded datasets & trained model weights
└── README.md            # Documentation
```

---

## 📦 Prerequisites & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UmiMeansSea/EcoAware-Classifier.git
   cd EcoAware-Classifier
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ installed. Install the required dependencies:
   ```bash
   pip install torch torchvision codecarbon
   ```

---

## ⚡ Usage

Run the classifier script to download the CIFAR-10 dataset (if not already downloaded), train the simple CNN baseline model, and track carbon emissions:

```bash
python eco_classifier.py
```

### Sample Output

```text
Downloading and preparing the training data...
Data is ready! We have 50000 training images.
Model built! It currently relies on heavy, 32-bit floating-point math.
Starting to train the heavy model. Tracking emissions...
Training finished!
Total Carbon Emissions: 0.12345 grams of CO2.
Saved the heavy 32-bit model to 'heavy_model.pth'
```

---

## 🔬 Next Steps / Roadmap

- [ ] Implement post-training quantization (FP32 to INT8) to compare memory reduction & inference speed.
- [ ] Measure emissions reduction after applying pruning and quantization techniques.
- [ ] Add evaluation metric reporting (Accuracy, F1-Score) alongside CO₂ metrics.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
