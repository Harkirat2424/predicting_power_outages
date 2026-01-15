# Predicting Power Outages from Severe Weather Events

A machine learning project that predicts power outages caused by severe weather events using XGBoost classification models. The project analyzes five types of severe weather events (Tornado, Thunderstorm, Hail, Heavy Snow, and High Wind) and their relationship with power outages across the United States from 2014-2023.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Code Workflow](#code-workflow)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)

## 🎯 Overview

This project develops predictive models to forecast power outages caused by severe weather events. By combining historical weather event data, real-time weather variables, and power outage records, the system can predict whether a severe weather event will result in power outages with high accuracy.

**Key Features:**
- Event-specific feature engineering for each weather type
- Separate XGBoost models optimized for each event type
- Spatiotemporal matching of weather events with power outages
- Comprehensive evaluation metrics and model persistence

## 📁 Project Structure

```
predicting_power_outages/
│
├── codes/                          # Jupyter notebooks and model files
│   ├── ml_code_final.ipynb        # Main ML training pipeline
│   ├── outage_matching_code.ipynb # Matches weather events with outages
│   ├── weather_var_extract.ipynb   # Extracts weather variables from API
│   ├── long_lat_extract.ipynb      # Extracts coordinates for counties
│   ├── complete_data.csv           # Combined dataset for all events
│   ├── xgb_*.pkl                   # Trained XGBoost models (5 models)
│   └── scaler_*.pkl                # StandardScaler objects (5 scalers)
│
├── ml_data/                        # Processed datasets
│   ├── complete_data.csv           # Final combined dataset
│   ├── *_outage_combined_2014_2023.csv  # Event-specific datasets (5 files)
│   └── complete_data.csv.zip       # Compressed dataset
│
└── presentation.pdf                # Project presentation
```

## 📊 Data Sources

### 1. **Weather Event Data**
- Source: NOAA Storm Events Database
- Events: Tornado, Thunderstorm Wind, Hail, Heavy Snow, High Wind
- Time Period: 2014-2023
- Fields: Event location (latitude/longitude), date/time, state, county

### 2. **Power Outage Data**
- Source: EAGLE-I (Environment for Analysis of Geo-Located Energy Information)
- Fields: State, county, outage start time
- Time Period: 2014-2023

### 3. **Weather Variables**
- Source: Open-Meteo Historical Weather API
- Variables: 25+ meteorological parameters including:
  - Temperature (2m, soil at multiple depths)
  - Precipitation (rain, snowfall, snow depth)
  - Wind (speed at 10m/100m, direction, gusts)
  - Pressure, humidity, cloud cover
  - Soil moisture at multiple depths

## 🔬 Methodology

### Data Processing Pipeline

1. **Weather Variable Extraction** (`weather_var_extract.ipynb`)
   - Fetches historical weather data from Open-Meteo API
   - Matches weather variables to each weather event by location and time
   - Includes checkpointing for large-scale data extraction

2. **Location Matching** (`long_lat_extract.ipynb`)
   - Geocodes county names to latitude/longitude coordinates
   - Uses geopy library with rate limiting

3. **Outage Matching** (`outage_matching_code.ipynb`)
   - Matches weather events with power outages using:
     - **Spatial matching**: Same state and county
     - **Temporal matching**: Outage occurs within 6 hours after weather event
   - Creates binary labels: `caused_power_outage` (1 if matched, 0 otherwise)
   - Processes data year-by-year for memory efficiency

4. **Model Training** (`ml_code_final.ipynb`)
   - Event-specific feature engineering
   - Data preprocessing and scaling
   - Model training with hyperparameter optimization
   - Model evaluation and persistence

### Feature Engineering

Each weather event type has custom feature engineering:

- **Tornado**: Wind shear, convective potential, extreme wind indicators, tornado severity index
- **Thunderstorm**: Wind shear, heavy precipitation indicators, convective potential, thunder severity
- **Hail**: Dew point depression, precipitation spikes, gust shear, hail severity
- **Heavy Snow**: Temperature anomalies, snow rate, wind power, low visibility indicators
- **High Wind**: Wind shear, direction shear, gust ratio, low pressure indicators

### Machine Learning Approach

- **Algorithm**: XGBoost Classifier
- **Feature Selection**: Recursive Feature Elimination (RFE) - selects top 15 features
- **Class Imbalance**: SMOTE oversampling + class weight balancing
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold cross-validation
- **Evaluation Metrics**: ROC-AUC, Precision, Recall, F1-Score, Confusion Matrix

## 💻 Code Workflow

### Step 1: Extract Weather Variables
```python
# Run weather_var_extract.ipynb
# Fetches weather data from Open-Meteo API for each event
```

### Step 2: Match with Power Outages
```python
# Run outage_matching_code.ipynb
# Matches weather events with outages based on location and time
```

### Step 3: Train Models
```python
# Run ml_code_final.ipynb
# Trains separate XGBoost models for each event type
# Saves models and scalers as .pkl files
```

## 📈 Results

### Model Performance (ROC-AUC Scores)

| Event Type    | ROC-AUC | Samples | Key Features |
|--------------|---------|---------|--------------|
| **Hail**     | 0.91    | 17,477  | Temperature, dew point, precipitation spikes |
| **Thunderstorm** | 0.87 | 32,446  | Precipitation, wind gusts, convective potential |
| **Tornado**  | 0.83    | 2,759   | Wind shear, convective potential, extreme wind |
| **Heavy Snow** | ~0.85 | ~12,000 | Temperature anomaly, snowfall, wind power |
| **High Wind** | ~0.85 | ~18,000 | Wind shear, gust ratio, pressure |

### Combined Model Performance
- **Combined ROC-AUC**: ~0.85-0.87
- **Overall Accuracy**: ~80%
- **F1-Score**: ~0.80

### Key Insights
- Hail events show the highest predictive accuracy
- Thunderstorm events have the largest sample size
- Event-specific feature engineering significantly improves model performance
- Spatiotemporal matching (6-hour window) effectively captures outage causality

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required Python packages (see below)

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib matplotlib seaborn geopy requests tqdm scipy
```

Or create a requirements file:

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `xgboost` - Gradient boosting classifier
- `imbalanced-learn` - SMOTE oversampling
- `joblib` - Model persistence
- `matplotlib` - Visualization
- `geopy` - Geocoding
- `requests` - API calls
- `tqdm` - Progress bars
- `scipy` - Spatial operations (KDTree)

## 📖 Usage

### Training Models

1. **Prepare Data**: Ensure you have the complete dataset in `codes/complete_data.csv`

2. **Run Training Notebook**:
   ```bash
   jupyter notebook codes/ml_code_final.ipynb
   ```
   This will:
   - Load and preprocess the data
   - Train models for each event type
   - Save models as `xgb_*.pkl` and scalers as `scaler_*.pkl`
   - Generate evaluation metrics and visualizations

### Making Predictions

```python
import joblib
import pandas as pd
import numpy as np

# Load model and scaler for a specific event type
event_type = 'thunderstorm'  # or 'tornado', 'hail', 'heavy_snow', 'high_wind'
model = joblib.load(f'codes/xgb_{event_type}.pkl')
scaler = joblib.load(f'codes/scaler_{event_type}.pkl')

# Prepare your data (must include all features used during training)
# See ml_code_final.ipynb for required features per event type
features = [...]  # Your feature array

# Scale and predict
features_scaled = scaler.transform([features])
probability = model.predict_proba(features_scaled)[0][1]
prediction = 1 if probability >= 0.4 else 0

print(f"Outage Probability: {probability:.2%}")
print(f"Prediction: {'Outage Likely' if prediction else 'No Outage'}")
```

### Data Processing Workflow

If you need to process raw data:

1. **Extract Weather Variables**: Run `weather_var_extract.ipynb`
   - Requires Open-Meteo API key
   - Processes events year-by-year with checkpointing

2. **Match Outages**: Run `outage_matching_code.ipynb`
   - Requires EAGLE-I outage data
   - Matches events with outages using 6-hour time window

3. **Combine Datasets**: The notebooks include code to combine yearly data into final datasets

## 🔍 Model Performance

### Evaluation Metrics

Each model is evaluated using:
- **ROC-AUC Score**: Area under the receiver operating characteristic curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification performance

### Feature Importance

The models use Recursive Feature Elimination to select the top 15 most important features for each event type. Key features typically include:
- Event-specific severity indices
- Location features (state, county, coordinates)
- Temporal features (month, hour)
- Core weather variables (temperature, precipitation, wind)

## 📝 Notes

- **Data Size**: The complete dataset contains ~293,000 records across all event types
- **Time Period**: Models trained on 2014-2023 data
- **Geographic Coverage**: United States (all states)
- **API Requirements**: Open-Meteo API key required for weather variable extraction
- **Memory Considerations**: Large datasets processed in chunks for efficiency

## 🤝 Contributing

This is a research project. For questions or improvements, please refer to the presentation document for detailed methodology and results.

## 📄 License

This project is for research and educational purposes.

---

**Author**: Harkirat Singh  
**Project**: Predicting Power Outages from Severe Weather Events  
**Year**: 2024

