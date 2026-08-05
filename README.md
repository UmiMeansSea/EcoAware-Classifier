# Frugal AI & Sobriété Numérique: Edge Image Classification

## Project Objective
This project demonstrates the principles of *Sobriété Numérique* (Digital Sobriety) by building a highly accurate, extremely lightweight Convolutional Neural Network (CNN) optimized for Edge CPU deployment. The goal was to drastically reduce the carbon footprint, memory size, and inference latency of computer vision models without sacrificing performance.

## The Engineering Pivot
Initially, I experimented with Post-Training Static Quantization (PTQ) on a pre-trained MobileNetV2. However, the depthwise separable convolutions collapsed under static 8-bit integer constraints, and Dynamic Quantization resulted in a 9MB file size. 

To achieve true Frugal AI, I refused the compromises of large pre-trained models and engineered a custom native architecture: **FrugalNet**.

## Results & Dashboard
By processing native 32x32 images and applying perfect 8-bit Static Quantization (INT8) using the FBGEMM backend, FrugalNet achieved a massive performance gain:
*   **Memory Footprint:** 276 KB (Reduced by ~97% from Baseline)
*   **Inference Latency:** 0.328 ms/img 
*   **Accuracy:** 62.51% (A ~20% absolute increase over the FP32 Baseline CNN)
*   **Carbon Emissions (Training):** ~1.38 grams of CO2

*(Insert your dashboard screenshot from the assets folder here!)*
`![Frugal AI Dashboard](assets/image_923820.png)`

##  Tech Stack & Evolution
This project bridges foundational data science with advanced model compression techniques:
*   **Core Data Analysis:** NumPy, Pandas, Scikit-Learn
*   **Deep Learning & Compression:** PyTorch, TorchVision, FBGEMM (Quantization)
*   **Tracking & Visualization:** CodeCarbon, Streamlit