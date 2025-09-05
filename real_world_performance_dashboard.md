# NFL Model Performance Dashboard
Generated: 2025-09-01 09:30:32

## Model Performance Summary
| Model | Accuracy | RMSE | ROI | Sharpe Ratio |
|-------|----------|------|-----|--------------|
| Real NFL Model | 18.6% | 16.22 | -64.4% | -13.74 |
| random | 16.9% | 15.55 | -67.6% | -14.98 |
| naive_mean | 25.4% | 15.12 | -51.4% | -9.82 |
| vegas_baseline | 18.6% | 15.33 | -64.4% | -13.74 |
| market_efficient | 20.3% | 15.50 | -61.2% | -12.63 |

### Real NFL Model Detailed Analysis

#### Core Performance
- **Accuracy**: 18.6%
- **Precision**: 100.0%
- **Recall**: 6.7%
- **F1-Score**: 0.125

#### Error Analysis
- **MAE**: 12.56 points
- **RMSE**: 16.22 points
- **MAPE**: 112.4%
- **Bias**: -2.98 points

#### Statistical Properties
- **Variance**: 254.27
- **Skewness**: -0.111
- **Kurtosis**: -0.006
- **Normality p-value**: 0.9416
- **Autocorrelation**: -0.031

#### Betting Performance
- **ROI**: -64.4%
- **Sharpe Ratio**: -13.74
- **Max Drawdown**: -0.0%
- **Hit Rate**: 18.6%

## Benchmark Comparisons
### Real NFL Model vs random
- **Overall Improvement**: -12.6%
- **Statistical Significance**: p < 0.050
- **Better at**: accuracy, precision, roi, sharpe_ratio
- **Worse at**: recall, f1_score, mae, rmse

### Real NFL Model vs naive_mean
- **Overall Improvement**: 3.1%
- **Statistical Significance**: p < 0.500
- **Better at**: precision, recall, f1_score
- **Worse at**: accuracy, mae, rmse, roi, sharpe_ratio

### Real NFL Model vs vegas_baseline
- **Overall Improvement**: -7.7%
- **Statistical Significance**: p < 0.050
- **Better at**: precision, sharpe_ratio
- **Worse at**: recall, f1_score, mae, rmse

### Real NFL Model vs market_efficient
- **Overall Improvement**: -13.4%
- **Statistical Significance**: p < 0.050
- **Better at**: precision
- **Worse at**: accuracy, recall, f1_score, mae, rmse, roi, sharpe_ratio

## Performance Interpretation
❌ **Poor Performance**: Model underperforms random selection
📉 **Unprofitable**: Model would lose money in betting scenarios
📊 **Poor Risk-Adjusted Returns**: Low Sharpe ratio