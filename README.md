# Autoencoder-Based Three-Factor Model for the U.S. Treasury Yield Curve

An adapted replication of [Suimon et al. (2020)](https://doi.org/10.3390/jrfm13040082), *“Autoencoder-Based Three-Factor Model for the Yield Curve of Japanese Government Bonds and a Trading Strategy”*, applied to U.S. Treasury yield curve data.

**Course:** IFTE0004 Financial Analytics and Machine Learning  
**Institution:** UCL Institute of Finance & Technology

---

## Overview

This project replicates the autoencoder yield-curve factor model originally developed for Japanese Government Bonds (JGBs) and applies it to U.S. Treasury yields. The implementation covers:

- **Factor extraction:** a 5→3→5 autoencoder for learning compact yield-curve representations.
- **Benchmark comparison:** PCA with three components as the linear benchmark, alongside 2-, 3-, and 4-node autoencoder specifications.
- **Trading strategy:** rolling-window relative-value signals based on reconstruction residuals.
- **Stress testing:** out-of-sample evaluation on the 2020–2023 period.
- **Robustness check:** extended nine-maturity specification from 3M to 20Y, including short-end Treasury bill maturities.

---

## Repository Structure

```text
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
    ├── table3_extended_comparison.csv
    ├── factor_correlations_ae3.csv
    ├── decoder_weights_ae3.csv
    ├── extended_specification_summary.csv
    └── extended_factor_correlations_ae3.csv
```

---

## Data

- **Dataset:** `USdataYC.csv`
- **Primary specification:** 2Y, 5Y, 7Y, 10Y, and 20Y U.S. Treasury yields
- **Extended specification:** 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, and 20Y
- **Training sample:** October 1993 to December 2019, resampled to weekly Friday observations
- **Stress sample:** January 2020 to December 2023, held out for out-of-sample evaluation

The original paper uses JGB data. This project is therefore an adapted methodological replication using the provided U.S. Treasury yield curve dataset.

---

## Methodology

The replication follows the core structure of Suimon et al. (2020):

1. Preprocess daily yield data and resample to weekly observations.
2. Standardise yields using a scaler fitted only on the training sample.
3. Estimate PCA with three components as a linear benchmark.
4. Train autoencoders with two, three, and four hidden nodes.
5. Evaluate reconstruction performance using RMSE and R².
6. Interpret latent factors using level, slope, and curvature proxies.
7. Implement a rolling-window trading strategy using reconstruction residuals as relative-value signals.
8. Run a stress-period check and an extended maturity-scope robustness test.

The main autoencoder architecture is adapted to the U.S. maturity set as:

```text
5 input maturities → 3-node tanh bottleneck → 5 reconstructed maturities
```

---

## Key Results

### Reconstruction Performance

| Model | Train RMSE (bps) | Train R² | Stress RMSE (bps) | Stress R² |
|---|---:|---:|---:|---:|
| PCA (3 factors) | 2.93 | 0.9997 | 4.15 | 0.9993 |
| AE (2 nodes) | 14.04 | 0.9946 | 26.88 | 0.9689 |
| AE (3 nodes) | 7.51 | 0.9985 | 19.96 | 0.9829 |
| AE (4 nodes) | 11.10 | 0.9966 | 29.26 | 0.9631 |

The three-node autoencoder is the best-performing autoencoder specification, supporting the paper’s argument that three latent factors provide a parsimonious representation of the yield curve. PCA remains the strongest reconstruction benchmark for the U.S. Treasury curve, suggesting that the dominant movements in this dataset are largely linear and low-dimensional.

### Trading Strategy Results

| Strategy | 2Y | 5Y | 7Y | 10Y | 20Y | Average |
|---|---:|---:|---:|---:|---:|---:|
| Autoencoder | -2.75 | -1.36 | 0.72 | 3.19 | 1.19 | 0.20 |
| Trend-Follow | 1.22 | 0.01 | 0.22 | 0.36 | -0.15 | 0.33 |
| Always Long | 0.90 | 0.92 | 0.98 | 0.97 | 1.15 | 0.98 |
| Always Short | -0.90 | -0.92 | -0.98 | -0.97 | -1.15 | -0.98 |

The autoencoder strategy produces positive average gains at the 10Y and 20Y maturities, broadly consistent with the original paper’s finding that the signal is more useful at longer maturities. However, it does not outperform the always-long benchmark in the U.S. sample, partly reflecting the broad secular decline in Treasury yields during much of the training period.

### Robustness Check

The extended nine-maturity specification confirms that the three-factor structure is not driven solely by the five paper-aligned maturities. Three PCA factors explain 99.92% of variation in the extended curve, although reconstruction errors increase when short-end Treasury bill maturities are included.

---

## Generated Output Files

Figures and tables are saved in the `yield_curve_outputs/` directory after running the script.

### Figures

- `fig_eda.png` — U.S. Treasury yield evolution and selected yield curves
- `fig_factor_analysis.png` — PCA loadings and autoencoder decoder coefficients
- `fig_reconstruction.png` — Actual versus reconstructed yield curves
- `fig_trading.png` — Cumulative trading performance
- `fig_residuals.png` — AE(3) reconstruction residuals used as valuation signals
- `fig_stress_factors.png` — AE(3) hidden factors during training and stress periods

### Tables

- `table1_reconstruction.csv` — PCA and autoencoder reconstruction performance
- `table2_trading_5y_1m.csv` — Trading strategy results for the representative specification
- `table3_extended_comparison.csv` — Primary versus extended maturity-scope comparison
- `factor_correlations_ae3.csv` — Correlations between AE(3) factors and financial proxies
- `decoder_weights_ae3.csv` — AE(3) decoder weights
- `extended_specification_summary.csv` — Extended maturity reconstruction metrics
- `extended_factor_correlations_ae3.csv` — Extended maturity factor correlations

---

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

Required packages:

```text
numpy
pandas
scikit-learn
tensorflow
matplotlib
```

---

## Usage

Run the full pipeline with:

```bash
python Autoencoder_yield_curve.py
```

All output figures and tables will be saved to:

```text
yield_curve_outputs/
```

To run the notebook version:

```bash
jupyter notebook Autoencoder_yield_curve.ipynb
```

---

## Key Functions

| Function | Description |
|---|---|
| `build_ae()` | Constructs the autoencoder with configurable hidden nodes |
| `run_ae()` | Trains the autoencoder and evaluates reconstruction performance |
| `run_pca()` | Fits the PCA benchmark and computes reconstruction metrics |
| `run_trading()` | Executes the rolling-window trading strategy |
| `run_extended_spec()` | Re-estimates the model on the extended maturity specification |

---

## Notes

This project is an adapted replication. The original paper uses Japanese Government Bond yields, while this implementation applies the methodology to U.S. Treasury yield curve data. Exact numerical replication is therefore not expected; the aim is to reproduce the methodology and assess whether the main findings transfer to a different sovereign bond market.

---

## References

- Suimon, Y., Sakaji, H., Izumi, K., & Matsushima, H. (2020). Autoencoder-based three-factor model for the yield curve of Japanese Government Bonds and a trading strategy. *Journal of Risk and Financial Management*, 13(4), 82.
```
