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
![[Alt text](https://github.com/viperapex/Crop-Export-Value-Prediction-using-Multilayer-Perceptron/tree/9150513787499ccfef32cf06888543a26c14aff2/content/data/output)]
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
The model provides future year forecasting capabilities for 2024-2026 across all countries in the dataset, generating country-specific predictions that are exported to CSV format for easy access and analysis. The system produces several key output files including the preprocessed and merged dataset (combined_data_all_years.csv), trained model weights (best_model.pth), and region-specific prediction files (predictions_{region}.csv). Key features encompass robust data handling with missing value treatment, temporal consistency maintenance, and automated feature scaling, combined with advanced model capabilities such as automated hyperparameter tuning, learning rate scheduling, early stopping based on validation loss, and comprehensive model evaluation metrics.


This project enables future year forecasting and country-specific predictions, with results exported to CSV format for practical use. From a technical perspective, the model utilizes synthetic data for future predictions when actual data is unavailable, applies log-transformations to monetary values to handle skewness, is optimized for Google Colab environment, and employs random seeds for reproducible results. The results interpretation shows decreasing loss curves indicating successful training, a high R-squared score demonstrating strong predictive capability, and closely aligned validation and test losses suggesting good generalization performance across different datasets.
