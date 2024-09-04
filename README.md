## Project Overview

This project, developed for the 2024 Emory Health AI Bias Datathon, explores the use of Grad-CAM with ResNet to interpret how a model identifies breast cancer in mammogram images. By visualizing the model's focus areas, we provide insights into potential biases within the Emory EMBED dataset and the model's decision-making process.

## Key Components

### 1. Dataset Curation - Masked ROI (`Dataset_Curation.ipynb`)
Curates mammogram images into positive and negative groups based on BIRADS scores and diagnostic outcomes. The curated groups are then exported as CSV files for analysis.

### 2. Interactive Grad-CAM Visualization (`demo_grad-cam_visualization.ipynb`)
Visualizes critical regions in images using Grad-CAM on a pre-trained ResNet50 model, providing an interactive exploration of model inferences.

### 3. Emory Mammo Metadata Exploration (`Emory_Mammo_Metadata_Exploration.ipynb`)
Explores and visualizes metadata from the Emory EMBED mammogram dataset, offering insights into data relationships.

### 4. ResNet and Grad-CAM (`ResNet and Gradcam.ipynb`)
Combines ResNet50 and Grad-CAM to interpret model predictions on mammogram images, evaluating model performance through metrics like accuracy and AUC.

## Getting Started

### Prerequisites
Ensure the following libraries are installed:

- TensorFlow
- NumPy
- Pandas
- Torch
- torchvision
- PIL (Pillow)
- Matplotlib
- CV2 (OpenCV)
- Requests
- tqdm
- IPython

Example import:

```python
import tensorflow as tf
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
```
## Setup

1. **Clone the Repository**
```bash
   git clone https://github.com/your-username/2024-Emory-Health-AI-Bias-Datathon.git
   cd 2024-Emory-Health-AI-Bias-Datathon
```

2. **Install Dependencies:**
```bash
  pip install tensorflow numpy pandas torch torchvision pillow matplotlib opencv-python requests tqdm ipython
```

3. **Run the Notebooks:**
- Launch Jupyter and open the desired notebook to reproduce the analyses.

# Results and Insights

- **Bias Detection:** Grad-CAM visualizations highlight potential biases in model predictions, offering a window into areas that may require further scrutiny.
- **Model Interpretation:** These visualizations enhance understanding of the ResNet model’s decision-making process, crucial for building trust in AI applications within healthcare.

# Contributions

This project was a collaborative effort during the 2024 Emory Health AI Bias Datathon. Special thanks to the organizers, mentors, and team members for their contributions.

# License

This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments

We acknowledge the Emory EMBED dataset and the support from Emory University School of Medicine and the AI Bias Datathon organizers. Thanks to the open-source community for the tools and libraries that made this project possible.
