# PiNN for Slope Stability

This project uses a **Physics-Informed Neural Network (PiNN)** to analyze slope stability. It combines physics with neural networks to predict slope deformation and failure.

## Overview

The PiNN models a 2D slope problem using physics equations (linear elastic and gravity). It aims to predict slope stability in a way similar to traditional methods like **Finite Element Method (FEM)**.

## Features
- Predicts slope stability using PiNN.
- Compares PiNN results with FEM.
- Shows displacement and deformation.
- Handles gravity loading.
- Exports displacement data for plotting.



## Model Details

The PiNN is trained using:
- **Physics equations**: 2D linear elastic 
- **Material parameters**:
  - Young's Modulus (E): 50000 kN/m^2   (for example) to compare with FEM from Plaxis2d
  - Poisson's Ratio (ν): 0.3
  - Unit Weight (γ): 18 kN/m³ (for example) to compare with FEM from Plaxis2d
- **Boundary Conditions**: Fixed displacements on the bottom, left, and right sides.
  ![Slope Stability Visualization - Dimension](Dimension_2.JPG)

## Result

![Slope Stability Visualization](Result.png)
![Slope Stability Visualization](Animate.gif)
![Slope Stability Visualization](FEM.png)
![Slope Stability Visualization](Absolute_error.png)
![Slope Stability Visualization](RMSE.png)
![Slope Stability Visualization](Heatmap_Grid_search.png)

## Installation

Install the necessary tools using:

```bash
pip install torch matplotlib numpy
pip install pandas
pip install pytorch
