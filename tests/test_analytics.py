import numpy as np
import pandas as pd
import pytest

from briefing import build_advisor_brief
from analytics import (
    PortfolioInputError,
    covariance_matrix,
    portfolio_metrics,
    proportional_alternative_allocation,
    scenario_impact,
    validate_assumptions,
    validate_weights,
)


ASSETS = ["Equities", "Fixed income", "Alternatives"]
RETURNS = pd.Series([0.08, 0.04, 0.06], index=ASSETS)
VOLS = pd.Series([0.20, 0.05, 0.12], index=ASSETS)
FEES = pd.Series([0.01, 0.005, 0.02], index=ASSETS)
CORR = pd.DataFrame([[1, 0, 0.2], [0, 1, 0.1], [0.2, 0.1, 1]], index=ASSETS, columns=ASSETS)
RF = 0.02


def test_weights_sum_to_100_and_reject_invalid_weights():
    assert validate_weights({"a": 0.6, "b": 0.4}).sum() == pytest.approx(1.0)
    with pytest.raises(PortfolioInputError):
        validate_weights({"a": 0.6, "b": 0.3})
    with pytest.raises(PortfolioInputError):
        validate_weights({"a": 1.1, "b": -0.1})


@pytest.mark.parametrize("alternative_weight", [0.0, 0.1, 0.2, 0.3])
def test_proportional_alternatives_funding_preserves_mix_and_100_percent_total(alternative_weight):
    result = proportional_alternative_allocation({"Equities": 0.6, "Fixed income": 0.4}, alternative_weight)
    assert result.to_dict() == pytest.approx(
        {"Equities": 0.6 * (1 - alternative_weight), "Fixed income": 0.4 * (1 - alternative_weight), "Alternatives": alternative_weight}
    )
    assert result.sum() == pytest.approx(1)
    with pytest.raises(PortfolioInputError):
        proportional_alternative_allocation({"Equities": 0.6, "Fixed income": 0.4}, 0.31)


def test_covariance_matrix_is_volatility_outer_product_times_correlation():
    cov = covariance_matrix(VOLS, CORR)
    expected = np.outer(VOLS.to_numpy(), VOLS.to_numpy()) * CORR.to_numpy()
    np.testing.assert_allclose(cov.to_numpy(), expected)
    assert cov.loc["Equities", "Equities"] == pytest.approx(0.04)


def test_expected_return_fee_drag_and_net_return_are_calculated_once():
    weights = {"Equities": 0.5, "Fixed income": 0.3, "Alternatives": 0.2}
    metrics = portfolio_metrics(weights, RETURNS, VOLS, FEES, CORR, RF)
    assert metrics.gross_return == pytest.approx(0.5 * 0.08 + 0.3 * 0.04 + 0.2 * 0.06)
    assert metrics.fee_drag == pytest.approx(0.5 * 0.01 + 0.3 * 0.005 + 0.2 * 0.02)
    assert metrics.net_return == pytest.approx(metrics.gross_return - metrics.fee_drag)


def test_volatility_sharpe_and_component_risk_contributions():
    weights = pd.Series({"Equities": 0.5, "Fixed income": 0.3, "Alternatives": 0.2})
    metrics = portfolio_metrics(weights, RETURNS, VOLS, FEES, CORR, RF)
    cov = covariance_matrix(VOLS, CORR).to_numpy()
    w = weights.to_numpy()
    sigma = np.sqrt(w @ cov @ w)
    assert metrics.volatility == pytest.approx(sigma)
    assert metrics.sharpe == pytest.approx((metrics.net_return - RF) / sigma)
    np.testing.assert_allclose(metrics.risk_contributions.to_numpy(), w * (cov @ w) / sigma)
    assert metrics.risk_contributions.sum() == pytest.approx(metrics.volatility)


def test_scenario_impact_is_weighted_simple_return_and_checks_invalid_shocks():
    impact = scenario_impact({"Equities": 0.6, "Fixed income": 0.4}, {"Equities": -0.2, "Fixed income": 0.05})
    assert impact == pytest.approx(-0.10)
    with pytest.raises(PortfolioInputError):
        scenario_impact({"Equities": 1.0}, {"Equities": -1.01})


def test_assumption_validation_rejects_asymmetric_wrong_diagonal_and_non_psd():
    validate_assumptions(RETURNS.to_dict(), VOLS.to_dict(), FEES.to_dict(), CORR)
    asymmetric = CORR.copy()
    asymmetric.loc["Equities", "Fixed income"] = 0.2
    with pytest.raises(PortfolioInputError, match="symmetric"):
        validate_assumptions(RETURNS.to_dict(), VOLS.to_dict(), FEES.to_dict(), asymmetric)
    wrong_diagonal = CORR.copy()
    wrong_diagonal.loc["Equities", "Equities"] = 0.9
    with pytest.raises(PortfolioInputError, match="unit diagonal"):
        validate_assumptions(RETURNS.to_dict(), VOLS.to_dict(), FEES.to_dict(), wrong_diagonal)
    impossible = pd.DataFrame([[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]], index=ASSETS, columns=ASSETS)
    with pytest.raises(PortfolioInputError, match="positive semidefinite"):
        validate_assumptions(RETURNS.to_dict(), VOLS.to_dict(), FEES.to_dict(), impossible)


def test_alternatives_can_increase_risk_and_reduce_expected_return():
    # Hypothetical risky alternative with high correlation to equity and a high fee.
    high_risk_corr = pd.DataFrame([[1, 0, 0.95], [0, 1, 0.0], [0.95, 0, 1]], index=ASSETS, columns=ASSETS)
    low_return = RETURNS.copy()
    low_return["Alternatives"] = -0.01
    high_vol = VOLS.copy()
    high_vol["Alternatives"] = 0.35
    traditional = {"Equities": 0.6, "Fixed income": 0.4, "Alternatives": 0.0}
    proposed = {"Equities": 0.54, "Fixed income": 0.36, "Alternatives": 0.10}
    before = portfolio_metrics(traditional, low_return, high_vol, FEES, high_risk_corr, RF)
    after = portfolio_metrics(proposed, low_return, high_vol, FEES, high_risk_corr, RF)
    assert after.volatility > before.volatility
    assert after.net_return < before.net_return


def test_alternatives_can_improve_expected_return_and_reduce_risk():
    favorable_returns = RETURNS.copy()
    favorable_returns["Alternatives"] = 0.10
    favorable_vols = VOLS.copy()
    favorable_vols["Alternatives"] = 0.05
    independent_corr = pd.DataFrame(np.eye(3), index=ASSETS, columns=ASSETS)
    favorable_fees = FEES.copy()
    favorable_fees["Alternatives"] = 0.001
    traditional = {"Equities": 0.6, "Fixed income": 0.4, "Alternatives": 0.0}
    proposed = {"Equities": 0.54, "Fixed income": 0.36, "Alternatives": 0.10}
    before = portfolio_metrics(traditional, favorable_returns, favorable_vols, favorable_fees, independent_corr, RF)
    after = portfolio_metrics(proposed, favorable_returns, favorable_vols, favorable_fees, independent_corr, RF)
    assert after.net_return > before.net_return
    assert after.volatility < before.volatility


def test_scenario_shock_comparison_uses_each_actual_weight_and_shock():
    shock = {"Equities": -0.25, "Fixed income": 0.03, "Alternatives": -0.12}
    traditional_impact = scenario_impact({"Equities": 0.6, "Fixed income": 0.4, "Alternatives": 0.0}, shock)
    proposed_impact = scenario_impact({"Equities": 0.54, "Fixed income": 0.36, "Alternatives": 0.10}, shock)
    assert traditional_impact == pytest.approx(0.6 * -0.25 + 0.4 * 0.03)
    assert proposed_impact == pytest.approx(0.54 * -0.25 + 0.36 * 0.03 + 0.10 * -0.12)
    assert proposed_impact - traditional_impact == pytest.approx(0.0018)


def test_alternatives_scenario_can_improve_or_worsen_the_portfolio():
    proposed = {"Equities": 0.54, "Fixed income": 0.36, "Alternatives": 0.10}
    traditional = {"Equities": 0.6, "Fixed income": 0.4, "Alternatives": 0.0}
    favorable = {"Equities": -0.25, "Fixed income": 0.03, "Alternatives": -0.05}
    adverse = {"Equities": -0.05, "Fixed income": 0.02, "Alternatives": -0.50}
    assert scenario_impact(proposed, favorable) > scenario_impact(traditional, favorable)
    assert scenario_impact(proposed, adverse) < scenario_impact(traditional, adverse)


def test_brief_matches_calculated_metrics_and_lists_complete_assumptions():
    app_returns = pd.Series([0.07, 0.035, 0.065], index=ASSETS)
    app_vols = pd.Series([0.16, 0.06, 0.12], index=ASSETS)
    app_fees = pd.Series([0.0015, 0.001, 0.015], index=ASSETS)
    app_correlations = pd.DataFrame([[1.0, -0.1, 0.25], [-0.1, 1.0, 0.05], [0.25, 0.05, 1.0]], index=ASSETS, columns=ASSETS)
    app_risk_free = 0.03
    starting = {"Equities": 0.6, "Fixed income": 0.4, "Alternatives": 0.0}
    proposed = {"Equities": 0.54, "Fixed income": 0.36, "Alternatives": 0.10}
    before = portfolio_metrics(starting, app_returns, app_vols, app_fees, app_correlations, app_risk_free)
    after = portfolio_metrics(proposed, app_returns, app_vols, app_fees, app_correlations, app_risk_free)
    shocks = pd.DataFrame(
        [[-0.25, 0.03, -0.12], [0.0, 0.0, 0.0]],
        index=["Equity selloff", "Custom shocks"], columns=ASSETS,
    )
    rows = []
    for name, shock in shocks.iterrows():
        b_gross = scenario_impact(starting, shock.to_dict())
        p_gross = scenario_impact(proposed, shock.to_dict())
        b_net, p_net = b_gross - before.fee_drag, p_gross - after.fee_drag
        rows.append({
            "Scenario": name,
            "Starting portfolio": b_net,
            "Proposed portfolio": p_net,
            "Change vs. starting": p_net - b_net,
        })
    results = pd.DataFrame(rows).set_index("Scenario")

    brief = build_advisor_brief(starting, proposed, before, after, app_returns, app_vols, app_fees, app_correlations, app_risk_free, shocks, results)
    assert "Expected annual gross return: 5.60% → 5.69%" in brief
    assert "Weighted annual fee drag: 0.13% → 0.27%" in brief
    assert "Expected annual net return: 5.47% → 5.42% (decreases by 0.05%)" in brief
    assert "Annualized volatility: 9.66% → 9.08% (decreases by 0.58%)" in brief
    assert "Risk-free rate: 3.00%" in brief
    assert "| Equities | 1.00 | -0.10 | 0.25 |" in brief
    assert "asset return shocks: Equities -25.00%, Fixed income 3.00%, Alternatives -12.00%" in brief
    assert "**Custom shocks** (asset return shocks: Equities 0.00%, Fixed income 0.00%, Alternatives 0.00%)" in brief
    assert "this result reflects one year of fees only" in brief
    assert "not investment advice" in brief


def test_invalid_assumptions_and_risk_free_input_are_rejected():
    negative_vol = VOLS.copy()
    negative_vol["Equities"] = -0.1
    with pytest.raises(PortfolioInputError, match="Volatilities"):
        portfolio_metrics({"Equities": 0.6, "Fixed income": 0.4, "Alternatives": 0}, RETURNS, negative_vol, FEES, CORR, RF)
    with pytest.raises(PortfolioInputError, match="Risk-free"):
        portfolio_metrics({"Equities": 0.6, "Fixed income": 0.4, "Alternatives": 0}, RETURNS, VOLS, FEES, CORR, float("nan"))
    impossible_return = RETURNS.copy()
    impossible_return["Equities"] = -1.01
    with pytest.raises(PortfolioInputError, match="below -100%"):
        portfolio_metrics({"Equities": 0.6, "Fixed income": 0.4, "Alternatives": 0}, impossible_return, VOLS, FEES, CORR, RF)


def test_sharpe_is_undefined_when_portfolio_has_zero_volatility():
    zero_vol = pd.Series(0.0, index=ASSETS)
    metrics = portfolio_metrics({"Equities": 0.6, "Fixed income": 0.4, "Alternatives": 0}, RETURNS, zero_vol, FEES, CORR, RF)
    assert metrics.volatility == pytest.approx(0.0)
    assert np.isnan(metrics.sharpe)
