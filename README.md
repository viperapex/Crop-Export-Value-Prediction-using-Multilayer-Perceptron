# Crop Export Value Prediction using Multilayer Perceptron

## Project Overview

This project implements a Multilayer Perceptron (MLP) neural network to predict crop export values using agricultural and economic indicators from FAOSTAT data. The model processes multiple datasets to forecast export values for different countries and future years.

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

## Data Preprocessing

### Key Steps:

1. **Data Loading & Integration**
   - Mount Google Drive for data access
   - Load multiple CSV files with consistent naming patterns
   - Handle mixed data types and warnings

2. **Data Cleaning & Transformation**
   - Ensure consistent data types for Area and Year columns
   - Handle year formats (e.g., "2020-2021" to 2020)
   - Add missing years to maintain temporal consistency

3. **Feature Engineering**
   - Aggregate values by Area and Year
   - Create log-transformed features for skewed distributions
   - Generate interaction terms (CPI x Emissions, Land Use x Temp Change)
   - One-hot encoding for categorical 'Area' variable

## Model Architecture

### Network Structure:
- Input Layer: 218 features
- Hidden Layers: [128, 64, 32, 16, 8, 4, 2] neurons with ReLU activation
- Output Layer: 1 neuron (regression output)
- Regularization: Dropout (p=0.3) and L2 regularization

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

### 1. Setup Environment

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Data Preparation
- Place all CSV files in the specified Google Drive folder
- Ensure proper file naming conventions

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

### Prediction Capabilities:
- Future year forecasting
- Country-specific predictions
- Export results to CSV format

## Technical Notes

- The model uses synthetic data for future predictions when actual data is unavailable
- All monetary values are log-transformed to handle skewness
- The code is optimized for Google Colab environment
- Random seeds are set for reproducible results

## Results Interpretation

- The decreasing loss curves indicate successful training
- High R-squared score demonstrates strong predictive capability
- Validation and test losses are closely aligned, suggesting good generalization

This implementation provides a robust framework for agricultural export value prediction using neural networks and can be easily adapted for other regression tasks in the agricultural domain.
