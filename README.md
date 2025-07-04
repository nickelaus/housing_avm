# Housing AVM / Hybrid AVM (Automated Valuation Model) Project
Estimate house prices using key features on based on comps - see if house is overpriced! 
Combines web scraping, data preprocessing, and machine learning to predict property values.

## Project Overview

This project consists of two main components:
1. **Web Scraping Module** (`redfin_listings_scraper_filter.py`) - Extracts property data from Redfin listings
2. **Machine Learning Model** (`house_value_model.py`) - Trains and evaluates property valuation models

## Features

### Web Scraping (`redfin_listings_scraper_filter.py`)
- Scrapes detailed property information from Redfin URLs
- Extracts features like basement details, garage spaces, storage structures
- Analyzes property descriptions for renovation indicators
- Handles rate limiting and retry logic for robust scraping
- Supports both full scraping and cleanup-only modes

### Machine Learning Model (`house_value_model.py`)
- Supports multiple algorithms: Linear Regression, Ridge Regression, Random Forest, XGBoost
- Implements log transformation for handling high variance in property prices
- Includes hyperparameter tuning for XGBoost and Ridge regression
- Provides comprehensive model evaluation metrics
- Features correlation-based feature selection

## Installation

### Prerequisites
- Python 3.8+
- Chrome browser (for web scraping)
- ChromeDriver (automatically managed by Selenium)

### Required Packages
```bash
pip install pandas numpy scikit-learn xgboost
pip install selenium beautifulsoup4 requests
pip install matplotlib tkinter
pip install openpyxl  # For Excel file support
```

## Usage

### 1. Web Scraping

#### Full Scraping Mode (Default)
```bash
python redfin_listings_scraper_filter.py
```
- Opens file dialog to select CSV file with Redfin URLs
- Scrapes property details for each listing
- Saves enriched data to `star_valley_hs_area_redfin_sold_enriched_cleaned.csv`

#### Cleanup Mode
```bash
python redfin_listings_scraper_filter.py --mode cleanup
```
- Processes existing `star_valley_hs_area_redfin_sold_enriched.csv`
- Applies data cleaning and feature engineering
- Saves cleaned data without scraping

### 2. Machine Learning Model

```bash
python house_value_model.py
```

#### Required CSV Columns
Your input CSV must contain:
- `BEDS` - Number of bedrooms
- `BATHS` - Number of bathrooms  
- `PRICE` - Sale price (target variable)
- `LOT SIZE` - Lot size in square feet
- `SQUARE FEET` - Living area
- `YEAR BUILT` - Construction year
- Additional features from scraping (basement, garage, etc.)

## Data Processing Pipeline

### 1. Data Cleaning
- Removes properties older than 1.5 years
- Handles missing values
- Converts date formats
- Calculates derived features (age, days since sold)

### 2. Feature Engineering
- **Basement_Level**: 0=None, 1=Partial, 2=Unfinished/Full, 3=Partially Finished, 4=Finished
- **Garage_Spaces**: Number of garage spaces
- **STORAGE_STRUCTURE**: Binary indicator for storage buildings
- **Recently_Renovated**: Binary indicator based on listing description
- **LotSize_Log**: Log-transformed lot size

### 3. Model Training
- Log transformation of target variable (price) to handle high variance
- Correlation-based feature selection
- Train/test split with evaluation on log scale
- Hyperparameter tuning for selected algorithms

## Model Performance

Expected R² scores on log-transformed data:
- **Linear Regression**: ~0.56
- **Random Forest**: ~0.52  
- **XGBoost**: ~0.63 (with tuning)
- **Ridge Regression**: ~0.50-0.60 (with tuning)

## Configuration Options

### Scraping Parameters
- `max_retries`: Number of retry attempts (default: 3)
- `base_delay`: Base delay between requests (default: 3 seconds)
- Headless Chrome with anti-detection features

### Model Parameters
- `correlation_threshold`: Feature selection threshold (default: 0.5)
- `tune_hyperparameters_xgb`: Enable XGBoost tuning
- `tune_hyperparameters_ridge`: Enable Ridge tuning
- `n_optuna_trials`: Number of optimization trials for XGBoost

## File Structure

```
hybrid_AVM/
├── redfin_listings_scraper_filter.py  # Web scraping module
├── house_value_model.py               # ML model training
├── README.md                          # This file
└── data/
    ├── input_data.csv                 # Raw Redfin export
    └── processed_data.csv             # Cleaned and enriched data
```

## Troubleshooting

### Common Issues

1. **SSL/Network Errors**: These are normal during scraping and handled by retry logic
2. **GPU Warnings**: Cosmetic warnings from headless Chrome, don't affect functionality
3. **Negative R² Values**: Indicates poor model performance; check if using log-scale evaluation
4. **Chrome Driver Issues**: Ensure Chrome browser is installed and up-to-date

### Performance Tips

- Use `--mode cleanup` for faster processing of already-scraped data
- Adjust delay parameters if getting blocked by rate limiting
- Monitor scraping progress and restart if needed
- Consider reducing dataset size for initial testing

## License

This project is for educational and research purposes. Ensure compliance with Redfin's terms of service when scraping data.
