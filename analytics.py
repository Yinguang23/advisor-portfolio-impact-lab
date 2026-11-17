"""Transparent portfolio analytics for the Portfolio Impact Simulator.

All returns, volatilities, fees, and scenarios are annualized decimal inputs.
The module deliberately contains no fund data or market-data assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


class PortfolioInputError(ValueError):
    """Raised when portfolio assumptions are internally inconsistent."""


@dataclass(frozen=True)
class PortfolioMetrics:
    gross_return: float
    fee_drag: float
    net_return: float
    volatility: float
    sharpe: float
    risk_contributions: pd.Series


def validate_weights(weights: Mapping[str, float], *, tolerance: float = 1e-8) -> pd.Series:
    """Return weights as a Series after checking finite, nonnegative, 100% total."""
    if not weights:
        raise PortfolioInputError("At least one asset weight is required.")
    result = pd.Series(weights, dtype=float)
    if not np.isfinite(result.to_numpy()).all():
        raise PortfolioInputError("Portfolio weights must be finite numbers.")
    if (result < 0).any():
        raise PortfolioInputError("Portfolio weights cannot be negative.")
    if not np.isclose(result.sum(), 1.0, atol=tolerance, rtol=0):
        raise PortfolioInputError(f"Portfolio weights must sum to 100% (received {result.sum():.4%}).")
    return result


def validate_assumptions(
    expected_returns: Mapping[str, float],
    volatilities: Mapping[str, float],
    fees: Mapping[str, float],
    correlations: pd.DataFrame,
    *,
    tolerance: float = 1e-8,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Validate aligned capital-market assumptions and a PSD correlation matrix."""
    assets = list(expected_returns)
    if not assets or set(assets) != set(volatilities) or set(assets) != set(fees):
        raise PortfolioInputError("Returns, volatilities, and fees must define the same assets.")
    if list(correlations.index) != assets or list(correlations.columns) != assets:
        raise PortfolioInputError("Correlation rows and columns must match the asset assumptions in the same order.")
    exp = pd.Series(expected_returns, dtype=float)
    vol = pd.Series(volatilities, dtype=float)
    fee = pd.Series(fees, dtype=float)
    matrix = correlations.astype(float)
    for label, series in (("Expected returns", exp), ("Volatilities", vol), ("Fees", fee)):
        if not np.isfinite(series.to_numpy()).all():
            raise PortfolioInputError(f"{label} must be finite numbers.")
        if label == "Expected returns" and (series < -1).any():
            raise PortfolioInputError("Expected returns cannot be below -100%.")
    if (vol < 0).any():
        raise PortfolioInputError("Volatilities cannot be negative.")
    if (fee < 0).any():
        raise PortfolioInputError("Fees cannot be negative.")
    arr = matrix.to_numpy()
    if not np.isfinite(arr).all() or (arr < -1 - tolerance).any() or (arr > 1 + tolerance).any():
        raise PortfolioInputError("Correlations must be finite values between -1 and 1.")
    if not np.allclose(arr, arr.T, atol=tolerance, rtol=0):
        raise PortfolioInputError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(arr), 1.0, atol=tolerance, rtol=0):
        raise PortfolioInputError("Correlation matrix must have a unit diagonal.")
    if np.linalg.eigvalsh((arr + arr.T) / 2).min() < -tolerance:
        raise PortfolioInputError("Correlation matrix must be positive semidefinite.")
    return exp, vol, fee, matrix


def covariance_matrix(volatilities: Mapping[str, float] | pd.Series, correlations: pd.DataFrame) -> pd.DataFrame:
    """Build annual covariance matrix: diag(sigma) @ correlation @ diag(sigma)."""
    vols = pd.Series(volatilities, dtype=float)
    if list(correlations.index) != list(vols.index) or list(correlations.columns) != list(vols.index):
        raise PortfolioInputError("Correlation dimensions must match volatility assets in the same order.")
    if not np.isfinite(vols.to_numpy()).all() or (vols < 0).any():
        raise PortfolioInputError("Volatilities must be finite and nonnegative.")
    corr = correlations.to_numpy(dtype=float)
    if not np.isfinite(corr).all():
        raise PortfolioInputError("Correlations must be finite.")
    cov = np.outer(vols.to_numpy(), vols.to_numpy()) * corr
    return pd.DataFrame(cov, index=vols.index, columns=vols.index)


def portfolio_metrics(
    weights: Mapping[str, float] | pd.Series,
    expected_returns: Mapping[str, float] | pd.Series,
    volatilities: Mapping[str, float] | pd.Series,
    fees: Mapping[str, float] | pd.Series,
    correlations: pd.DataFrame,
    risk_free_rate: float,
) -> PortfolioMetrics:
    """Calculate expected gross/net return, fee drag, volatility, Sharpe, and risk.

    Fees are a weighted annual expense deducted exactly once from expected return.
    Volatility is calculated from gross return covariance (constant fees add no
    modeled volatility); Sharpe uses net expected return less the risk-free rate.
    """
    w = validate_weights(pd.Series(weights, dtype=float).to_dict())
    exp, vol, fee, corr = validate_assumptions(
        pd.Series(expected_returns, dtype=float).to_dict(),
        pd.Series(volatilities, dtype=float).to_dict(),
        pd.Series(fees, dtype=float).to_dict(),
        correlations,
    )
    if list(w.index) != list(exp.index):
        raise PortfolioInputError("Portfolio weights must match the assumption assets in the same order.")
    if not np.isfinite(risk_free_rate):
        raise PortfolioInputError("Risk-free rate must be finite.")
    cov = covariance_matrix(vol, corr).to_numpy()
    wv = w.to_numpy()
    sigma = float(np.sqrt(max(float(wv @ cov @ wv), 0.0)))
    gross = float(wv @ exp.to_numpy())
    fee_drag = float(wv @ fee.to_numpy())
    net = gross - fee_drag
    marginal = cov @ wv
    contributions = pd.Series((wv * marginal) / sigma if sigma > 0 else np.zeros_like(wv), index=w.index, name="Risk contribution")
    sharpe = (net - float(risk_free_rate)) / sigma if sigma > 0 else float("nan")
    return PortfolioMetrics(gross, fee_drag, net, sigma, float(sharpe), contributions)


def scenario_impact(weights: Mapping[str, float] | pd.Series, shocks: Mapping[str, float] | pd.Series) -> float:
    """Return the one-period portfolio impact for asset-class simple-return shocks."""
    w = validate_weights(pd.Series(weights, dtype=float).to_dict())
    shock = pd.Series(shocks, dtype=float)
    if list(shock.index) != list(w.index):
        raise PortfolioInputError("Scenario shocks must match portfolio assets in the same order.")
    if not np.isfinite(shock.to_numpy()).all() or (shock < -1).any():
        raise PortfolioInputError("Scenario shocks must be finite and cannot be below -100%.")
    return float(w.to_numpy() @ shock.to_numpy())


def proportional_alternative_allocation(
    traditional_weights: Mapping[str, float], alternative_weight: float,
    equity: str = "Equities", fixed_income: str = "Fixed income", alternative: str = "Alternatives",
) -> pd.Series:
    """Fund an alternatives sleeve proportionally from the traditional mix."""
    base = validate_weights(traditional_weights)
    if not 0 <= alternative_weight <= 0.30:
        raise PortfolioInputError("Alternative allocation must be between 0% and 30%.")
    if equity not in base or fixed_income not in base or alternative in base:
        raise PortfolioInputError("Traditional allocation must contain equities and fixed income only.")
    total_traditional = float(base[equity] + base[fixed_income])
    if total_traditional <= 0:
        raise PortfolioInputError("Traditional allocation must have positive equity and fixed-income weights.")
    remaining = 1.0 - alternative_weight
    out = pd.Series(0.0, index=[equity, fixed_income, alternative])
    out[equity] = base[equity] / total_traditional * remaining
    out[fixed_income] = base[fixed_income] / total_traditional * remaining
    out[alternative] = alternative_weight
    return validate_weights(out.to_dict())
