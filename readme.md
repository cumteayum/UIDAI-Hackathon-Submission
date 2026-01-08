# UIDAI Enrolment Analysis & Forecasting System

## Overview

This project implements an industry-grade, explainable machine learning system to forecast Aadhaar enrolment volumes across Indian states and districts.

The solution leverages:

- Gradient-boosted decision trees (CatBoost)
- Robust temporal feature engineering
- Automated administrative name harmonization
- Rolling-origin backtesting
- Uncertainty-aware multi-horizon forecasting

## Key Results

- Reduced categorical sparsity by ~22% via automated state/district harmonization
- Achieved up to **35% RMSE reduction** compared to naive lag baselines
- Stable multi-state forecasts with early overfitting detection
- Consistent MAE below 3 for mature months

## Modeling Approach

- Target: Daily total enrolments
- Loss Function: Poisson (count-aware)
- Features:
  - Short-term momentum (lag_1, lag_7, rolling_7)
  - Demographic ratios
  - Cyclical temporal encodings
  - High-cardinality administrative identifiers

## Validation Strategy

- Rolling monthly backtesting
- Error decomposition by geography
- Zero-safe MAPE computation
- Stability index monitoring

## Forecasting

- Recursive 2-year daily forecasting
- State-wise aggregation
- Confidence bands for uncertainty communication

## Explainability

- Built-in CatBoost feature importance
- Transparent driver analysis
- Logged model fingerprinting for reproducibility

## Outputs

- `model_log.txt` – complete training & evaluation trace
- `forecast_2yr.csv` – daily & monthly projections
- `report.txt` – exhaustive EDA & metric analysis
