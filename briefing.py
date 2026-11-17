"""Pure text rendering for the downloadable advisor briefing."""
from __future__ import annotations

from datetime import date
from typing import Mapping

import pandas as pd

from analytics import PortfolioMetrics


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _sharpe(value: float) -> str:
    return f"{value:.2f}" if pd.notna(value) else "N/M (zero volatility)"


def build_advisor_brief(
    starting_weights: Mapping[str, float],
    proposed_weights: Mapping[str, float],
    before: PortfolioMetrics,
    after: PortfolioMetrics,
    expected_returns: Mapping[str, float],
    volatilities: Mapping[str, float],
    fees: Mapping[str, float],
    correlations: pd.DataFrame,
    risk_free_rate: float,
    scenario_shocks: pd.DataFrame,
    scenario_results: pd.DataFrame,
    *,
    prepared: date | None = None,
) -> str:
    """Create a briefing that records all inputs and corresponding modeled results."""
    assets = list(expected_returns.keys())
    start_text = ", ".join(f"{asset.lower()} {_pct(starting_weights[asset])}" for asset in assets)
    proposed_text = ", ".join(f"{asset.lower()} {_pct(proposed_weights[asset])}" for asset in assets)
    return_delta = after.net_return - before.net_return
    risk_delta = after.volatility - before.volatility
    return_direction = "increases" if return_delta > 0 else "decreases" if return_delta < 0 else "is unchanged"
    risk_direction = "increases" if risk_delta > 0 else "decreases" if risk_delta < 0 else "is unchanged"

    correlation_rows = ["| Asset | " + " | ".join(assets) + " |", "|---|" + "---:|" * len(assets)]
    for asset in assets:
        correlation_rows.append("| " + asset + " | " + " | ".join(f"{correlations.loc[asset, col]:.2f}" for col in assets) + " |")

    scenario_lines = []
    for name, shocks in scenario_shocks.iterrows():
        result = scenario_results.loc[name]
        shock_text = ", ".join(f"{asset} {_pct(shocks[asset])}" for asset in assets)
        zero_shock_note = " No asset shocks were specified, so this result reflects one year of fees only." if (shocks == 0).all() else ""
        scenario_lines.append(
            f"- **{name}** (asset return shocks: {shock_text}): starting net "
            f"{result['Starting portfolio']:.2%}; proposed {result['Proposed portfolio']:.2%}; "
            f"difference {result['Change vs. starting']:+.2%}.{zero_shock_note}"
        )

    assumptions = (
        f"Expected annual gross returns: {', '.join(f'{a} {_pct(expected_returns[a])}' for a in assets)}. "
        f"Annualized volatilities: {', '.join(f'{a} {_pct(volatilities[a])}' for a in assets)}. "
        f"Annual management fees: {', '.join(f'{a} {_pct(fees[a])}' for a in assets)}. "
        f"Risk-free rate: {_pct(risk_free_rate)}."
    )
    return f"""# Advisor Portfolio Impact Brief

**Prepared:** {(prepared or date.today()).isoformat()}

**Purpose:** Illustrative comparison using editable hypothetical assumptions.

## Allocation
- Starting allocation (default 60/40): {start_text}.
- Proposed allocation: {proposed_text}.

## Modeled changes
- Expected annual gross return: {_pct(before.gross_return)} → {_pct(after.gross_return)}.
- Weighted annual fee drag: {_pct(before.fee_drag)} → {_pct(after.fee_drag)}.
- Expected annual net return: {_pct(before.net_return)} → {_pct(after.net_return)} ({return_direction} by {_pct(abs(return_delta))}).
- Annualized volatility: {_pct(before.volatility)} → {_pct(after.volatility)} ({risk_direction} by {_pct(abs(risk_delta))}).
- Net Sharpe ratio: {_sharpe(before.sharpe)} → {_sharpe(after.sharpe)}.

## Correlation assumptions
{chr(10).join(correlation_rows)}

## Diversification implications
The modeled annualized volatility {risk_direction} by {_pct(abs(risk_delta))}, from {_pct(before.volatility)} to {_pct(after.volatility)}. This change is driven by the selected asset volatilities and correlations; it is sensitive to those assumptions and does not guarantee diversification or downside protection.

## Scenario analysis
Scenarios are hypothetical direct asset-class return shocks, not forecasts or historical backtests. Net figures deduct one year of weighted annual fees.
{chr(10).join(scenario_lines)}

## Assumptions and limitations
{assumptions}

Fees reduce expected return once; volatility uses the covariance of asset returns and does not add constant fees as risk. The model omits taxes, transaction costs, liquidity, valuation lags, leverage, path dependence, rebalancing and investment-specific terms. All strategy assumptions are fictional and hypothetical. This brief is not investment advice, a forecast, a backtest, actual fund performance, or an offer. Diversification does not guarantee a profit or prevent loss.
"""
