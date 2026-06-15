# project_orchestration_ml

Project to predict used car prices and identify optimal investment opportunities.

## Context & Problematic

**What is the price of a car?**

This project aims to build an end-to-end Machine Learning pipeline to:
1. **Predict** the fair market price of a used car (using XGBoost/LightGBM/RF).
2. **Identify** undervalued cars (where listed price < predicted price) to find the best ROI.
3. **Orchestrate** the workflow using MLflow (tracking), FastAPI (API serving), and Streamlit (user interface), all managed via Docker and a Makefile.

## Installation

```bash
make install
```

## data

For data you must download :
https://www.kaggle.com/datasets/sharmajicoder/used-car-price-prediction-dataset

And extract : 
used_car_price_prediction_1M.csv
in the data folder : data/used_car_price_prediction_1M.csv