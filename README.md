# Fine-Grained Vehicle Classification: ResNet50 vs EfficientNet-B4

This repository contains the code and final report for an academic project exploring Fine-Grained Visual Categorization (FGVC) on the Stanford Cars dataset.

## Overview

The goal of this project is to accurately classify 196 specific vehicle classes. Since FGVC tasks suffer from low inter-class variance (cars look alike) and high intra-class variance (different angles/lighting), we employed a custom **letterboxing** pre-processing technique to preserve geometric aspect ratios.

We implemented a two-phase transfer learning approach using PyTorch to compare two models:
* **ResNet50:** Achieved 80.59% test accuracy (batch size 32).
* **EfficientNet-B4:** Achieved 82.39% test accuracy (batch size 8).

Additionally, using **Grad-CAM** interpretability techniques, we discovered an unexpected behavior where the CNNs attempted to exploit the zero-padding from our letterboxing step to deduce the vehicle's original aspect ratio. This phenomenon aligns with recent literature on zero-padding artifacts in CNNs.

## Repository Structure
* `notebook.ipynb`: The main Jupyter Notebook containing the data loading, letterboxing pre-processing, training pipelines, and Grad-CAM visualization code.
* `requirements.txt`: Python dependencies required to run the notebook.
* `report/`: Contains the LaTeX source code, image assets, and final compiled PDF of the academic paper detailing our methodology and findings.

## Dataset
This project uses the official Stanford Cars Dataset. The dataset is quite large and is not included in this repository.

You can find more information about the dataset and download it from [Papers with Code: Stanford Cars](https://paperswithcode.com/dataset/stanford-cars).

## Requirements & Setup

1. Clone the repository.
2. Create a virtual environment (optional but recommended).
3. Install the dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset and place it in a `data/stanford_cars/` folder in the root directory.
5. Run all cells in `notebook.ipynb`.
