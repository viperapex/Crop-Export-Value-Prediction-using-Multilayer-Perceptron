# Crop Export Value Prediction using Multilayer Perceptron

## Project Overview

This project develops a comprehensive machine learning system for predicting agricultural export values using a Multilayer Perceptron (MLP) neural network. The model leverages extensive datasets from FAOSTAT (Food and Agriculture Organization of the United Nations) to forecast crop export values across different countries and time periods. By integrating multiple dimensions of agricultural and economic data, this system provides valuable insights for stakeholders in the agricultural sector.

The primary objective is to create a robust predictive model that can assist governments, agricultural businesses, policymakers, and international organizations in making informed decisions about crop production, trade policies, and resource allocation. The model analyzes historical patterns and relationships between various economic, environmental, and agricultural factors to generate accurate export value predictions.

## Dataset Description

The project uses 13 different FAOSTAT datasets:
- Consumer prices indicators
- Crops production indicators
- Emissions data
- Employment statistics
- Exchange rates
- Fertilizers use
- Food security indicators
- Food balances indicators
- Food trade indicators (Target variable)
- Foreign direct investment
- Land temperature change
- Land use
- Pesticides use

## Model Architecture

### Network Structure:
 Input Layer: 218 features
 Hidden Layers: [128, 64, 32, 16, 8, 4, 2] neurons with ReLU activation
 Output Layer: 1 neuron (regression output)
 Regularization: Dropout (p=0.3) and L2 regularization

### Training Configuration:
- Optimizer: Adam (learning rate=0.001, weight decay=1e-5)
- Loss Function: Mean Squared Error (MSE)
- Scheduler: ReduceLROnPlateau
- Batch Size: 128
- Epochs: 120
- Validation: 20% of training data

## Model Performance

- Final Test Loss: 3.4642
- R-squared Score: 0.9259
- The model shows excellent predictive performance with 92.59% variance explained

## Usage Instructions

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```
Place all CSV files in the specified Google Drive folder and also Ensure proper file naming conventions

### 3. Model Training

The code automatically:
- Preprocesses all datasets
- Trains the MLP model with early stopping
- Saves the best performing model
- Generates loss plots

### 4. Making Predictions

```python
# Change target region as needed
target_region = 'Area_Albania'  # Modify this for different countries
```

The model can predict export values for future years (2024-2026) for any country in the dataset.

## Output Files

- `combined_data_all_years.csv`: Preprocessed and merged dataset
- `best_model.pth`: Trained PyTorch model weights
- `predictions_{region}.csv`: Prediction results for specific regions

## Key Features

### Data Handling:
- Robust missing value handling
- Temporal consistency across datasets
- Automated feature scaling

### Model Features:
- Automated hyperparameter tuning
- Learning rate scheduling
- Early stopping based on validation loss
- Comprehensive model evaluation

