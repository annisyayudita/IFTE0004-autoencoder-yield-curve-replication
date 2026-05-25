# Autoencoder-Based Three-Factor Model for the U.S. Treasury Yield Curve

An adapted replication of [Suimon et al. (2020)](https://doi.org/10.3390/jrfm13040082) — *"Autoencoder-Based Three-Factor Model for the Yield Curve of Japanese Government Bonds and a Trading Strategy"* — applied to U.S. Treasury data.

**Course:** IFTE0004 Financial Analytics and Machine Learning  
**Institution:** UCL Institute of Finance & Technology

---

## Overview

This project replicates the autoencoder yield-curve factor model originally developed for Japanese Government Bonds (JGBs) and applies it to U.S. Treasury yields. The implementation covers:

- **Factor extraction:** A 5→3→5 autoencoder that recovers Level, Slope, and Curvature factors from the U.S. yield curve
- **Benchmark comparison:** PCA (3 components) as the linear benchmark, with 2/3/4-node autoencoder architectural comparison
- **Trading strategy:** Rolling-window relative-value signals using reconstruction residuals
- **Stress testing:** Out-of-sample evaluation on the 2020–2023 period (COVID, inflation, Fed tightening)
- **Robustness:** Extended 9-maturity specification (3M–20Y) including T-bills

## Repository Structure

```
├── Autoencoder_yield_curve.py        
├── Autoencoder_yield_curve.ipynb     
├── USdataYC.csv                      
├── requirements.txt                 
├── README.md
└── yield_curve_outputs/
    ├── fig_eda.png
    ├── fig_factor_analysis.png
    ├── fig_reconstruction.png
    ├── fig_residuals.png
    ├── fig_trading.png
    ├── fig_stress_factors.png
    ├── table1_reconstruction.csv
    ├── table2_trading_5y_1m.csv
    └── table3_extended_comparison.csv
```

## Data

- **Source:** Provided dataset (`USdataYC.csv`)
- **Primary specification:** 5 maturities (2Y, 5Y, 7Y, 10Y, 20Y)
- **Extended specification:** 9 maturities (3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y)
- **Training period:** October 1993 – December 2019 (1,371 weekly observations)
- **Stress period:** January 2020 – December 2023 (209 observations, held-out)

## Key Results

| Model | Train RMSE (bps) | Train R² | Stress RMSE (bps) | Stress R² |
|-------|-----------------|----------|-------------------|----------|
| PCA (3 factors) | 2.93 | 0.9998 | 4.15 | 0.9993 |
| AE (3 nodes) | 7.51 | 0.9985 | 19.96 | 0.9829 |

- Three-node autoencoder confirmed as optimal (consistent with original paper)
- Decoder weights recover Level, Slope, and Curvature structure
- Trading signal generates positive returns at 10Y (+3.19 bp/month) and 20Y (+1.19 bp/month)

## Generated Figures

### Yield Curve Evolution
![EDA](yield_curve_outputs/fig_eda.png)

### Factor Analysis: PCA Loadings & AE Decoder Weights
![Factor Analysis](yield_curve_outputs/fig_factor_analysis.png)

### Actual vs Reconstructed Yield Curves
![Reconstruction](yield_curve_outputs/fig_reconstruction.png)

### Cumulative Trading Performance
![Trading](yield_curve_outputs/fig_trading.png)

### AE(3) Reconstruction Residuals
![Residuals](yield_curve_outputs/fig_residuals.png)

### Stress-Period Factor Continuity
![Stress Factors](yield_curve_outputs/fig_stress_factors.png)

## Requirements

```
numpy
pandas
scikit-learn
tensorflow
matplotlib
```

## Usage

**Run the full pipeline:**
```bash
python Autoencoder_yield_curve.py
```

All figures and tables are saved to the `yield_curve_outputs/` directory. Set `RUN_FULL_TRADING_GRID = True` at the top of the script to evaluate all learning window and horizon combinations.

**Jupyter notebook:**
```bash
jupyter notebook Autoencoder_yield_curve.ipynb
```

## Key Functions

| Function | Description |
|----------|-------------|
| `build_ae()` | Constructs the autoencoder with configurable hidden nodes |
| `run_ae()` | Trains the autoencoder with early stopping and evaluates reconstruction |
| `run_pca()` | Fits PCA benchmark and computes reconstruction metrics |
| `run_trading()` | Executes rolling-window trading strategy with annual retraining |
| `run_extended_spec()` | Re-estimates on the 9-maturity extended specification |

## References

- Suimon, Y., Sakaji, H., Izumi, K., & Matsushima, H. (2020). Autoencoder-based three-factor model for the yield curve of Japanese Government Bonds and a trading strategy. *Journal of Risk and Financial Management*, 13(4), 82.
- Baldi, P. & Hornik, K. (1989). Neural networks and principal component analysis. *Neural Networks*, 2(1), 53–58.
- Nelson, C.R. & Siegel, A.F. (1987). Parsimonious modeling of yield curves. *Journal of Business*, 60(4), 473–489.

