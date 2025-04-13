# Install libraries without pmdarima to avoid binary incompatibility
!pip install numpy pandas matplotlib statsmodels scipy scikit-learn arch --quiet

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from arch import arch_model
import itertools
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Set random seed for reproducibility
np.random.seed(42)

# -----------------------------------
# Step 1: Generate Enhanced Synthetic Data
# -----------------------------------
# Create 5 years of monthly sales data (60 months)
dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='MS')
n = len(dates)

# Synthetic sales: trend + growing seasonality + noise + random spikes (in £)
trend = np.linspace(4000, 12000, n)  # Linear trend from £4K to £12K
seasonality = (1000 + 0.2 * np.arange(n)) * np.sin(2 * np.pi * np.arange(n) / 12)  # Growing seasonality
noise = np.random.normal(0, 400, n)  # Random noise
sales = trend + seasonality + noise

# Add random spikes (e.g., promotional events)
spikes = np.random.choice(n, size=3, replace=False)
sales[spikes] += np.random.normal(3000, 500, 3)

# Create DataFrame
data = pd.DataFrame({'Date': dates, 'Sales': sales})
data.set_index('Date', inplace=True)

# Plot synthetic data
plt.figure(figsize=(12, 6))
plt.plot(data['Sales'], label='Synthetic Sales')
plt.title('Enhanced Synthetic Monthly Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales (£)')
plt.legend()
plt.savefig('/content/synthetic_sales.png')
plt.close()

# -----------------------------------
# Step 2: Exploratory Data Analysis
# -----------------------------------
# Stationarity tests
def adf_test(series, title=''):
    result = adfuller(series.dropna())
    print(f'ADF Test - {title}')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    print()

def kpss_test(series, title=''):
    stat, p_value, lags, crit = kpss(series.dropna())
    print(f'KPSS Test - {title}')
    print('KPSS Statistic:', stat)
    print('p-value:', p_value)
    print('Critical Values:', crit)
    print()

adf_test(data['Sales'], 'Original Series')
kpss_test(data['Sales'], 'Original Series')

# Decompose time series
decomposition = seasonal_decompose(data['Sales'], model='additive', period=12)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.savefig('/content/decomposition.png')
plt.close()

# -----------------------------------
# Step 3: Data Preprocessing
# -----------------------------------
# First and seasonal differencing
data['Sales_diff'] = data['Sales'].diff()
data['Sales_seasonal_diff'] = data['Sales'].diff(12)

# Test stationarity
adf_test(data['Sales_diff'], 'First Differenced Series')
kpss_test(data['Sales_diff'], 'First Differenced Series')
adf_test(data['Sales_seasonal_diff'], 'Seasonal Differenced Series')
kpss_test(data['Sales_seasonal_diff'], 'Seasonal Differenced Series')

# ACF and PACF plots
plt.figure(figsize=(12, 4))
plt.subplot(121)
plot_acf(data['Sales_diff'].dropna(), lags=20, ax=plt.gca())
plt.title('ACF Plot (First Difference)')
plt.subplot(122)
plot_pacf(data['Sales_diff'].dropna(), lags=20, ax=plt.gca())
plt.title('PACF Plot (First Difference)')
plt.savefig('/content/acf_pacf.png')
plt.close()

# -----------------------------------
# Step 4: Model Selection (Grid Search)
# -----------------------------------
# Refined grid search for SARIMA parameters
p = q = range(0, 2)
P = Q = range(0, 2)
d = [1]  # First differencing confirmed by ADF/KPSS
D = [1]  # Seasonal differencing confirmed
m = 12  # Seasonal period (monthly)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], m) for x in itertools.product(P, D, Q)]

best_aic = float('inf')
best_params = None
best_model = None

print('Running SARIMA grid search...')
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(data['Sales'],
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = (param, param_seasonal)
                best_model = results
            print(f'Tried SARIMA{param}x{param_seasonal}, AIC: {results.aic:.2f}')
        except Exception as e:
            print(f'Skipped SARIMA{param}x{param_seasonal} due to error: {e}')
            continue

if best_params is None:
    raise ValueError("No valid SARIMA model found. Check data or parameters.")

print('Best SARIMA Parameters:', best_params)
print('Best AIC:', best_aic)

# Fit the best model
model = SARIMAX(data['Sales'],
                order=best_params[0],
                seasonal_order=best_params[1],
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit(disp=False)

# Save model summary
with open('/content/model_summary.txt', 'w') as f:
    f.write(str(results.summary()))

# Plot actual vs fitted
plt.figure(figsize=(12, 6))
plt.plot(data['Sales'], label='Actual Sales')
plt.plot(results.fittedvalues, label='Fitted Values', linestyle='--')
plt.title('Actual vs Fitted Sales')
plt.xlabel('Date')
plt.ylabel('Sales (£)')
plt.legend()
plt.savefig('/content/actual_vs_fitted.png')
plt.close()

# -----------------------------------
# Step 5: Advanced Residual Diagnostics
# -----------------------------------
# Residual analysis
residuals = results.resid
plt.figure(figsize=(12, 6))
plt.subplot(211)
residuals.plot(title='Residuals')
plt.ylabel('Residuals (£)')
plt.subplot(212)
plot_acf(residuals, lags=20, ax=plt.gca())
plt.title('ACF of Residuals')
plt.tight_layout()
plt.savefig('/content/residuals.png')
plt.close()

# Q-Q plot for residual normality
plt.figure(figsize=(8, 6))
qqplot(residuals, line='s', fit=True, ax=plt.gca())
plt.title('Q-Q Plot of Residuals')
plt.savefig('/content/qqplot.png')
plt.close()

# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print('Ljung-Box Test:')
print(lb_test)

# Shapiro-Wilk test
stat, p = shapiro(residuals)
print('Shapiro-Wilk Test p-value:', p)

# ARCH test for heteroscedasticity
arch_test = arch_model(residuals, vol='ARCH').fit(disp='off')
print('ARCH Test Summary:')
print(arch_test.summary())

# -----------------------------------
# Step 6: Forecasting with Uncertainty
# -----------------------------------
# Forecast for 12 months
forecast = results.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Probabilistic forecast via simulation
simulated_results = results.simulate(anchor='end', nsimulations=12, repetitions=1000)
simulated_ci = np.quantile(simulated_results, [0.025, 0.975], axis=1).T

# Create forecast DataFrame
forecast_index = pd.date_range(start='2025-01-01', periods=12, freq='MS')
forecast_df = pd.DataFrame({'Forecast': forecast_mean,
                            'Lower CI': forecast_ci.iloc[:, 0],
                            'Upper CI': forecast_ci.iloc[:, 1],
                            'Simulated Lower CI': simulated_ci[:, 0],
                            'Simulated Upper CI': simulated_ci[:, 1]},
                           index=forecast_index)

# Save forecast to CSV
forecast_df.to_csv('/content/sales_forecast_2025.csv')

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(data['Sales'], label='Historical Sales')
plt.plot(forecast_df['Forecast'], label='Forecast', color='red')
plt.fill_between(forecast_df.index,
                 forecast_df['Lower CI'],
                 forecast_df['Upper CI'],
                 color='red', alpha=0.1, label='95% CI')
plt.fill_between(forecast_df.index,
                 forecast_df['Simulated Lower CI'],
                 forecast_df['Simulated Upper CI'],
                 color='blue', alpha=0.05, label='Simulated 95% CI')
plt.title('Sales Forecast for 2025')
plt.xlabel('Date')
plt.ylabel('Sales (£)')
plt.legend()
plt.savefig('/content/forecast.png')
plt.close()

# Display forecast
print('\nSales Forecast for 2025 (£):')
print(forecast_df)

# -----------------------------------
# Step 7: Enhanced Model Evaluation
# -----------------------------------
# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=3, test_size=12)  # Fits 60 samples
rmse_scores = []
mae_scores = []
mape_scores = []

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

for train_idx, test_idx in tscv.split(data['Sales']):
    train = data['Sales'].iloc[train_idx]
    test = data['Sales'].iloc[test_idx]
    
    if len(train) < 24 or len(test) == 0:
        print(f"Skipping fold: train_size={len(train)}, test_size={len(test)}")
        continue
    
    try:
        model_cv = SARIMAX(train,
                           order=best_params[0],
                           seasonal_order=best_params[1],
                           enforce_stationarity=False,
                           enforce_invertibility=False)
        results_cv = model_cv.fit(disp=False)
    except Exception as e:
        print(f"Skipping fold due to model error: {e}")
        continue
    
    forecast_cv = results_cv.get_forecast(steps=len(test))
    forecast_mean_cv = forecast_cv.predicted_mean
    
    if len(forecast_mean_cv) != len(test):
        print(f"Skipping fold: forecast_size={len(forecast_mean_cv)}, test_size={len(test)}")
        continue
    
    rmse_scores.append(np.sqrt(mean_squared_error(test, forecast_mean_cv)))
    mae_scores.append(mean_absolute_error(test, forecast_mean_cv))
    mape_scores.append(mape(test, forecast_mean_cv))

print('\nCross-Validation Metrics:')
if rmse_scores:
    print(f'Average RMSE: £{np.mean(rmse_scores):.2f} (±{np.std(rmse_scores):.2f})')
    print(f'Average MAE: £{np.mean(mae_scores):.2f} (±{np.std(mae_scores):.2f})')
    print(f'Average MAPE: {np.mean(mape_scores):.2f}% (±{np.std(mape_scores):.2f})')
else:
    print('No valid folds for cross-validation metrics.')

# Single train-test split for baseline comparison
train = data['Sales'][:-12]
test = data['Sales'][-12:]

if len(test) == 0:
    raise ValueError("Test set is empty. Check data splitting.")

model = SARIMAX(train,
                order=best_params[0],
                seasonal_order=best_params[1],
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit(disp=False)

forecast_test = results.get_forecast(steps=12)
forecast_mean_test = forecast_test.predicted_mean

if len(forecast_mean_test) != len(test):
    raise ValueError(f"Forecast length ({len(forecast_mean_test)}) does not match test length ({len(test)}).")

rmse = np.sqrt(mean_squared_error(test, forecast_mean_test))
mae = mean_absolute_error(test, forecast_mean_test)
mape_val = mape(test, forecast_mean_test)

naive_forecast = train[-12:].values
if len(naive_forecast) != len(test):
    raise ValueError(f"Naive forecast length ({len(naive_forecast)}) does not match test length ({len(test)}).")

naive_rmse = np.sqrt(mean_squared_error(test, naive_forecast))
naive_mae = mean_absolute_error(test, naive_forecast)
naive_mape = mape(test, naive_forecast)

print('\nSingle Train-Test Split Metrics:')
print(f'SARIMA RMSE: £{rmse:.2f}, MAE: £{mae:.2f}, MAPE: {mape_val:.2f}%')
print(f'Seasonal Naive RMSE: £{naive_rmse:.2f}, MAE: £{naive_mae:.2f}, MAPE: {naive_mape:.2f}%')

# -----------------------------------
# Step 8: Advanced Visualizations
# -----------------------------------
# 1. Forecast Error Distribution
errors = test - forecast_mean_test
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=20, density=True, alpha=0.7, color='blue')
plt.title('Forecast Error Distribution')
plt.xlabel('Error (£)')
plt.ylabel('Density')
plt.savefig('/content/error_distribution.png')
plt.close()

# 2. Cumulative Forecast vs Actual
cumulative_actual = test.cumsum()
cumulative_forecast = forecast_mean_test.cumsum()
plt.figure(figsize=(12, 6))
plt.plot(test.index, cumulative_actual, label='Cumulative Actual Sales', color='blue')
plt.plot(test.index, cumulative_forecast, label='Cumulative Forecast Sales', color='red', linestyle='--')
plt.title('Cumulative Sales: Actual vs Forecast')
plt.xlabel('Date')
plt.ylabel('Cumulative Sales (£)')
plt.legend()
plt.savefig('/content/cumulative_forecast.png')
plt.close()

# -----------------------------------
# Step 9: Save Summary Report
# -----------------------------------
with open('/content/forecast_summary.txt', 'w') as f:
    f.write('Enhanced Business Sales Forecasting Summary\n')
    f.write('=======================================\n')
    f.write(f'Best SARIMA Model: {best_params}\n')
    f.write(f'AIC: {best_aic:.2f}\n')
    f.write('\nCross-Validation Metrics:\n')
    if rmse_scores:
        f.write(f'Average RMSE: £{np.mean(rmse_scores):.2f} (±{np.std(rmse_scores):.2f})\n')
        f.write(f'Average MAE: £{np.mean(mae_scores):.2f} (±{np.std(mae_scores):.2f})\n')
        f.write(f'Average MAPE: {np.mean(mape_scores):.2f}% (±{np.std(mape_scores):.2f})\n')
    else:
        f.write('No valid folds for cross-validation metrics.\n')
    f.write('\nSingle Train-Test Split Metrics:\n')
    f.write(f'SARIMA RMSE: £{rmse:.2f}, MAE: £{mae:.2f}, MAPE: {mape_val:.2f}%\n')
    f.write(f'Seasonal Naive RMSE: £{naive_rmse:.2f}, MAE: £{naive_mae:.2f}, MAPE: {naive_mape:.2f}%\n')
    f.write('\nLjung-Box Test (lag 10):\n')
    f.write(str(lb_test))
    f.write(f'\nShapiro-Wilk Test p-value: {p:.4f}\n')
    f.write('\nForecast for 2025 (£):\n')
    f.write(str(forecast_df))

# -----------------------------------
# README 
# -----------------------------------
readme_content = """
Business Sales Forecasting Model

Overview
--------
This notebook implements a robust time series forecasting model to predict monthly sales (in £) for a retail business using synthetic data. The model leverages SARIMA with grid search for parameter selection, cross-validation, probabilistic forecasts, and advanced diagnostics, making it suitable for business applications and portfolio showcasing.

Features
--------
- Synthetic Data: 5 years of monthly sales with trend, growing seasonality, noise, and promotional spikes.
- EDA: Stationarity tests (ADF, KPSS), decomposition, ACF/PACF plots.
- Model: SARIMA with grid search for optimal parameters.
- Diagnostics: Residual analysis (Q-Q plot, Ljung-Box, Shapiro-Wilk, ARCH test).
- Forecasting: 12-month forecast for 2025 with standard and simulated confidence intervals.
- Evaluation: Cross-validation (RMSE, MAE, MAPE) and comparison to a seasonal naive baseline.
- Visualizations: Error distribution, cumulative sales, and actual vs fitted plots.

Setup
-----
- Environment: Google Colab (Python 3).
- Libraries: Installed automatically (`numpy`, `pandas`, `matplotlib`, `statsmodels`, `scipy`, `scikit-learn`, `arch`).
- Execution: Run all cells sequentially.

Outputs
-------
- Plots: Saved in `/content/` (e.g., `synthetic_sales.png`, `forecast.png`, `actual_vs_fitted.png`).
- Files:
  - `model_summary.txt`: SARIMA model details.
  - `sales_forecast_2025.csv`: Forecast table.
  - `forecast_summary.txt`: Comprehensive summary with metrics and diagnostics.
- Console: Displays diagnostics, metrics, and forecast.

Usage
-----
1. Run the notebook in Google Colab.
2. Inspect plots and console outputs.
3. Download files from `/content/` for reporting (e.g., via Colab’s file explorer).
4. Modify synthetic data parameters (e.g., trend, spikes) for experimentation.

Notes
-----
- Currency is British Pounds (£) for UK context.
- Code is modular and extensible (e.g., add exogenous variables for SARIMAX).
- Optimized for Colab with minimal setup, avoiding pmdarima to prevent binary incompatibility.
- Cross-validation errors may be high due to promotional spikes; consider outlier handling for improved metrics.

with open('/content/README.md', 'w') as f:
    f.write(readme_content)