# Advisor Portfolio Impact Lab

A standalone Streamlit Portfolio Impact Simulator for exploring how a hypothetical alternatives allocation may affect a traditional portfolio. All capital-market assumptions are editable examples; no actual fund data is used.

## Run locally

Requires Python 3.12.

```bash
cd advisor-portfolio-impact-lab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`).

## Deployment configuration

The project is structured for a Streamlit Community Cloud deployment from the project root with `app.py` as the entrypoint. Install the packages listed in `requirements.txt`. The same files can be committed to a GitHub repository; no credentials or secrets are required by this illustrative app. The deployment path has not been exercised as part of local QA.

## Run the tests

```bash
python -m pytest -q
```

## Model conventions

- The default starting allocation is 60% equities and 40% fixed income. Alternatives can be set from 0% to 30%; the default funding method takes the allocation proportionally from equities and fixed income. A custom mode lets the user set equity and alternatives weights, with fixed income as the remainder.
- Expected return, volatility, fees, and correlations are hypothetical annual assumptions. The correlation matrix must be symmetric, have a unit diagonal, and be positive semidefinite.
- Gross expected portfolio return is the weighted sum of asset expected returns. Weighted annual management fees are deducted once to obtain expected net return. Fees do not add volatility in this constant-fee model.
- Volatility uses the covariance matrix `diag(volatility) × correlation × diag(volatility)`. Sharpe uses expected net return less the risk-free rate. Component risk contributions use `weight × marginal contribution / portfolio volatility` and sum to total volatility.
- Scenario results apply the specified simple return shocks to weights. Net scenario impact subtracts one year of weighted annual fees.

## Important limitation

Illustrative only; not investment advice, a forecast, a backtest, actual fund performance, or an offer. This simplified model omits, among other things, taxes, transaction costs, leverage, liquidity, valuation lags, rebalancing, path dependence, and investment-specific terms. Diversification does not guarantee a profit or protect against loss.
