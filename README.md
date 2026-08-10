# EcoAware-Classifier 🌿

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21874631.svg)](https://doi.org/10.5281/zenodo.21874631)

> **Official codebase for the paper:** *Frugal AI: Sustainable Deep Learning Compression via Pruning and Mixed-Precision QAT.*

This repository contains the complete end-to-end pipeline for optimizing a Convolutional Neural Network (CNN) under strict environmental constraints. By combining iterative magnitude pruning with mixed-precision Quantization-Aware Training (QAT), this project achieves extreme edge-deployment efficiency without sacrificing model integrity.

### 🚀 Key Performance Metrics
* **Compression Ratio:** 3.73x reduction in binary file size (from 2085.93 KB down to 559.53 KB).
* **Accuracy Retention:** Maintained test accuracy within a 0.67% margin of the FP32 baseline.
* **Carbon Footprint:** The entire end-to-end optimization pipeline emitted just **12.006 g CO2e**, tracked empirically via CodeCarbon.

### 📖 Read the Paper
The full research preprint is permanently available on Zenodo. If you use this code or methodology in your own research, please cite it as:

```bibtex
@misc{karjee2026frugalai,
  author       = {Karjee, Aniket},
  title        = {Frugal AI: Sustainable Deep Learning Compression via Pruning and Mixed-Precision QAT},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21874631},
  url          = {https://doi.org/10.5281/zenodo.21874631}
}
```

---

# Project Report: Frugal AI – Sustainable Deep Learning Compression via Pruning and Mixed-Precision QAT

## 1. Executive Summary

As deep learning models grow in size and computational demand, the environmental impact of training and deploying these models has become a critical concern. This project explores the principles of **Sobriété Numérique (Digital Sobriety)** by optimizing a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.

The objective was to dramatically reduce the model's memory footprint and operational carbon emissions without sacrificing predictive power. Through a custom pipeline integrating **Iterative Magnitude Pruning (70% sparsity)** and **Mixed-Precision Quantization-Aware Training (QAT)**, the final model achieved a **3.73× reduction in file size** with only a negligible **0.67% drop in accuracy**, all while maintaining a highly sustainable carbon footprint of **12.006 g CO₂e** per training cycle.

---

## 2. Methodology & Architecture

### 2.1 The Baseline Model (FP32)

A lightweight custom architecture (`SimpleCNN`) was developed to establish a healthy baseline. To prevent overfitting and ensure robust feature learning before compression, the following regularizations were applied:

* **Architecture:** 2 Convolutional Layers (with BatchNorm and ReLU) + 2 Fully Connected Layers.
* **Regularization:** `Dropout(0.5)` in the classifier head and L2 Weight Decay (`1e-4`).
* **Optimization:** Stochastic Gradient Descent (SGD) with Nesterov momentum and Cosine Annealing learning rate schedules.
* **Early Stopping:** Automated halting based on validation loss to prevent wasted compute.

### 2.2 Iterative L1 Magnitude Pruning

To reduce mathematical redundancy, the baseline model was subjected to iterative unstructured magnitude pruning.

* **Process:** Weights closest to zero were masked in five scheduled steps (20% → 40% → 55% → 65% → 70%).
* **Healing:** After each pruning step, the active weights were fine-tuned for 4-5 epochs to adapt to the newly introduced sparsity.
* **Result:** A fully functional model where 70% of the network parameters were permanently set to exactly zero.

### 2.3 Mixed-Precision Quantization-Aware Training

Standard Post-Training Quantization (PTQ) caused severe accuracy degradation (quantization shock) on the 70% sparse model. To recover this loss, a custom Eager-Mode QAT pipeline was engineered:

* **Fake-Quantization:** Observer nodes were inserted during training to allow the sparse weights to mathematically adapt to the rigid INT8 grid.
* **Mixed-Precision Bypass (Sensitivity Trap Avoidance):** The extreme edges of the network (the first convolutional layer and the final classification logits) are highly sensitive to rounding errors. These layers were explicitly skipped and kept in FP32, while the heavy inner layers (`conv2` and `fc1`) were compressed to INT8.
* **Dropout Management:** Dropout was explicitly frozen (`eval()` mode) during QAT to prevent activation observers from capturing skewed, corrupted distributions.

### 2.4 Environmental Auditing (CodeCarbon)

The entire three-phase pipeline was wrapped in the `CodeCarbon` API to track the exact hardware energy consumption and convert it into carbon equivalent emissions (CO₂e) based on the local energy grid.

---

## 3. Results & Performance Metrics

The compression pipeline successfully condensed the model while protecting its feature-extraction capabilities.

| Metric | FP32 Baseline | INT8 QAT (Final) | Delta / Ratio |
| --- | --- | --- | --- |
| **Test Accuracy** | 76.15% | 75.48% | **-0.67%** (End-to-End Loss) |
| **Model Size** | 2085.93 KB | 559.53 KB | **3.73×** Compression |
| **Sparsity** | 0% | 70% | N/A |
| **Carbon Footprint** | N/A | 12.0064 g CO₂e | Highly Sustainable |

**Key Findings:**

1. **Pruning Tolerance:** The model tolerated a 70% loss of weights exceptionally well, dropping only 1.49% accuracy prior to quantization.
2. **QAT Recovery:** Actively training the network with fake-quantization nodes actually *boosted* the pruned model's accuracy by +0.82%, proving that weights can successfully re-align to 8-bit boundaries if given the chance to adapt.

---

## 4. Dashboard & Visual Analytics

To make the carbon tracking transparent and actionable, a custom interactive Streamlit dashboard was built to visualize the environmental cost of algorithmic iterations.

**Environmental Audit — CodeCarbon Footprint**
The dashboard aggregates emissions across all recorded training sessions:

* The total cumulative emissions across 11 tracked experiment runs stand at 53.4401 g CO₂e.
* Total energy consumed throughout the project is 0.074905 kWh.
* The total training duration across all experiments is 8929.5 seconds.
* The carbon tracking bar chart reveals that early scripts emitted 0.219 g CO₂e, while the final `frugal_pipeline` runs stacked progressively higher, reaching 12.006 g CO₂e for the final optimized execution.

**Full Model Comparison: Legacy vs. Pipeline**
The side-by-side performance charts highlight the vast improvements of the newly engineered pipeline over the legacy approach:

* **Accuracy:** The Legacy Baseline achieved a top-1 accuracy of 36.13%. The new Pipeline FP32 Baseline achieved a best accuracy of 76.15%, representing a massive +40.02% improvement over the legacy model.
* **Size Optimization:** The Legacy Baseline had a file size of 1052.9 KB. The new pipeline's smallest compressed size is 559.5 KB, which is a reduction of 493.4 KB compared to the legacy baseline.
* **Inference Latency:** Across all evaluated models, latency remains highly constrained, ranging from 0.202 ms/img to a maximum of 0.225 ms/img.

**Pipeline Deep-Dive: FP32 → Pruned → INT8**
The deep-dive section illustrates the exact cost of each compression phase:

* Moving from the Pipeline FP32 Baseline (76.15%) to the Pipeline Pruned 70% model (74.66%) resulted in a minor accuracy drop of -1.49%.
* The model size remained constant at roughly 2086 KB between the FP32 Baseline and the Pruned FP32 model, before dropping drastically to 560 KB following INT8 quantization.
* The final quantized INT8 model recovered accuracy slightly to finish at 75.48%.

---

## 5. Conclusion

This project proves that edge-deployable deep learning does not require sacrificing massive amounts of accuracy or burning excessive compute power. By combining Iterative Pruning with Mixed-Precision Quantization-Aware Training, it is possible to achieve near-native FP32 performance within an INT8 footprint. Implementing continuous environmental auditing via CodeCarbon ensures that the pursuit of algorithmic efficiency remains aligned with the core principles of digital sobriety.

---

## 6. How to Run

To replicate the pipeline, train the models, and view the environmental dashboard, follow these steps in your terminal:

**Step 1: Execute the Training & Compression Pipeline**
Run the primary script to generate the FP32, pruned, and quantized models. CodeCarbon will automatically track this execution and append the emissions data to `emissions.csv`.

```bash
python frugal_pipeline.py
```

**Step 2: Run the Benchmark Suite**
Evaluate all generated models against the CIFAR-10 test set. This script will generate the latency and accuracy metrics and save them to `all_models_comparison.json`.

```bash
python evaluate_all.py
```

**Step 3: Launch the Visual Dashboard**
Start the Streamlit application to visualize the environmental audit, model sizes, and accuracy charts directly in your web browser.

```bash
streamlit run dashboard.py
```