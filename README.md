# 🏠 House Price Prediction

A machine learning project that predicts house prices from historical housing data and
presents the results through an interactive dashboard.

## 📌 Overview

This project trains a regression model on a housing dataset (`house_price_data.csv`)
and exposes the analysis through a browser-based dashboard. Model evaluation metrics
and outputs are stored in `model_results.json`, while visual analysis is saved as
`ml_analysis.png`.

## ✨ Features

- Data loading, cleaning, and feature engineering from `house_price_data.csv`
- Machine learning regression model for house price prediction
- Model evaluation with results exported to `model_results.json`
- Exploratory data analysis plots saved as `ml_analysis.png`
- Interactive dashboard (`run_dashboard.py`) with custom styling (`style.html`)
- Frontend data files (`data.js`, `dashboard_data.js`) for chart rendering

## 🗂️ Project Structure
House_price_prediction/
├── main.py # Core ML pipeline: data processing, training, evaluation
├── run_dashboard.py # Launches the interactive dashboard
├── house_price_data.csv # Raw housing dataset
├── model_results.json # Saved model metrics / results
├── ml_analysis.png # ML analysis visualization
├── style.html # Dashboard styling
├── data.js # Frontend data for dashboard charts
├── dashboard_data.js # Dashboard-specific data
└── requirements.txt # Python dependencies
