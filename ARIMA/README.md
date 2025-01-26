# ARIMA Model Explanation

## 1. ARIMA Model Explanation

ARIMA (AutoRegressive Integrated Moving Average) is a popular statistical method used for time series forecasting. It combines three components:

### 1.1. AR (AutoRegressive) Process
The autoregressive part of ARIMA is modeled as:
\[
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p} + \epsilon_t
\]
Where:
- \(X_t\) is the value at time \(t\).
- \(\phi_1, \phi_2, \dots, \phi_p\) are the parameters (coefficients) of the AR model.
- \(p\) is the order of the autoregressive model.
- \(\epsilon_t\) is the white noise error term at time \(t\).

### 1.2. I (Integrated) Process
The integration part is used to make the time series stationary. It involves differencing the data:
\[
Y_t = X_t - X_{t-1}
\]
Where:
- \(Y_t\) is the differenced series.
- \(X_t\) and \(X_{t-1}\) are the values at time \(t\) and \(t-1\), respectively.

This process is repeated \(d\) times (where \(d\) is the differencing order) until the series becomes stationary.

### 1.3. MA (Moving Average) Process
The moving average part models the relationship between the current value and the previous error terms. The formula is:
\[
X_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}
\]
Where:
- \(X_t\) is the value at time \(t\).
- \(\mu\) is the mean of the series (or the intercept term).
- \(\epsilon_t, \epsilon_{t-1}, \dots, \epsilon_{t-q}\) are the white noise error terms.
- \(\theta_1, \theta_2, \dots, \theta_q\) are the parameters (coefficients) of the MA model.
- \(q\) is the order of the moving average model.

### 1.4. ARIMA Model Formula
Combining the three components, the ARIMA model is defined as:
\[
(1 - \phi_1 B - \phi_2 B^2 - \dots - \phi_p B^p)(1 - B)^d X_t = \epsilon_t + \theta_1 B \epsilon_{t-1} + \theta_2 B \epsilon_{t-2} + \dots + \theta_q B \epsilon_{t-q}
\]
Where:
- \(B\) is the backshift operator: \(B X_t = X_{t-1}\).
- \(d\) is the order of differencing.
- \(p\) and \(q\) are the autoregressive and moving average orders, respectively.

### Key Points:
- **AR** (AutoRegressive) part deals with the relationship between the current value and its past values.
- **I** (Integrated) part makes the data stationary by differencing.
- **MA** (Moving Average) part deals with the relationship between the current value and the past errors.

In practice, you choose the parameters \(p\), \(d\), and \(q\) based on the data and methods such as AIC (Akaike Information Criterion) for model selection.

## 2. Conclusion

This explanation covers the core concepts and mathematical foundation of the ARIMA model, which can be applied for time series forecasting. The model's performance can be improved by fine-tuning the parameters \(p\), \(d\), and \(q\) based on the characteristics of the data.

## Results
### Best Model
The best ARIMA model based on the stepwise AIC minimization process is:
- **ARIMA(2, 1, 3)(0, 0, 0)[0] intercept**

### Mean Squared Error
The Mean Squared Error (MSE) for the final model is approximately **14.1**.

## Warnings
The following warning was generated due to deprecations in `scikit-learn`:

```
 FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
```
This is a non-critical warning and can be ignored or suppressed using the code below:
```
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```
