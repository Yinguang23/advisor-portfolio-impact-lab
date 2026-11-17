from __future__ import annotations

import inspect
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    PortfolioInputError,
    portfolio_metrics,
    proportional_alternative_allocation,
    scenario_impact,
    validate_assumptions,
    validate_weights,
)
from briefing import build_advisor_brief


st.set_page_config(page_title="Portfolio Impact Simulator", page_icon="◈", layout="wide")

NAVY = "#11263D"
BLUE = "#3C76A5"
PALE_BLUE = "#EAF2F8"
TEAL = "#327F7D"
GOLD = "#C6953E"
GRAY = "#7C8997"
GREEN = "#237A58"
RED = "#B34D4D"
ASSETS = ["Equities", "Fixed income", "Alternatives"]


def stretch_kwargs(function):
    """Use current width API when available, while retaining older Streamlit support."""
    parameters = inspect.signature(function).parameters
    return {"width": "stretch"} if "width" in parameters else {"use_container_width": True}


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}"


def sharpe_text(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "N/M"


def metric_table(before, after):
    return pd.DataFrame(
        {
            "Starting portfolio": [before.gross_return, before.fee_drag, before.net_return, before.volatility, before.sharpe],
            "Proposed portfolio": [after.gross_return, after.fee_drag, after.net_return, after.volatility, after.sharpe],
        },
        index=["Expected gross return", "Annual fee drag", "Expected net return", "Annualized volatility", "Sharpe ratio (net)"],
    )


def comparison_card(title: str, before: str, after: str, change: str, favorable: bool | None) -> str:
    status = "positive" if favorable is True else "negative" if favorable is False else "neutral"
    direction = "Improves" if favorable is True else "Deteriorates" if favorable is False else "Unchanged"
    return (
        f'<section class="compare-card"><div class="compare-title">{escape(title)}</div>'
        f'<div class="compare-values"><div><span>Starting</span><strong>{escape(before)}</strong></div>'
        f'<div class="proposed-value"><span>Proposed</span><strong>{escape(after)}</strong></div></div>'
        f'<div class="change {status}">{direction} · {escape(change)}</div></section>'
    )


@st.cache_data
def defaults():
    # Fictional annual assumptions for illustration, not actual fund results.
    expected = pd.Series({"Equities": 0.070, "Fixed income": 0.035, "Alternatives": 0.065})
    vol = pd.Series({"Equities": 0.160, "Fixed income": 0.060, "Alternatives": 0.120})
    fees = pd.Series({"Equities": 0.0015, "Fixed income": 0.0010, "Alternatives": 0.0150})
    corr = pd.DataFrame(
        [[1.00, -0.10, 0.25], [-0.10, 1.00, 0.05], [0.25, 0.05, 1.00]], index=ASSETS, columns=ASSETS
    )
    return expected, vol, fees, corr


st.markdown(
    """
    <style>
    :root { --navy:#11263D; --ink:#20374C; --muted:#687C8F; --blue:#3C76A5; --line:#DFE7EE; }
    html, body, [class*="css"] { font-family: Inter, "Aptos", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .stApp { background:#F4F7FA; color:var(--ink); }
    h1,h2,h3,h4 { color:var(--navy); letter-spacing:-.025em; }
    h1 { font-size:2.15rem !important; font-weight:750 !important; }
    h2 { font-size:1.45rem !important; font-weight:700 !important; }
    h3 { font-size:1.12rem !important; font-weight:700 !important; }
    .block-container { max-width:1500px; padding-top:1.25rem; padding-bottom:2.5rem; }
    [data-testid="stSidebar"] { background:#EDF2F6; border-right:1px solid #DCE5EC; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:var(--navy); }
    .hero { background:linear-gradient(112deg,#11263D 0%,#193A56 66%,#28536F 100%); color:#F7FAFC;
        padding:1.35rem 1.65rem; border-radius:15px; margin:0 0 .85rem; }
    .hero h1 { color:white !important; margin:.25rem 0 .2rem; }
    .hero p { color:#DCE7EF; margin:0; font-size:.98rem; }
    .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:750; color:#B9D1E1; }
    .disclaimer { background:#FFFDF7; border:1px solid #E9DFC8; border-left:4px solid #C6953E; border-radius:8px;
        padding:.63rem .9rem; margin:.65rem 0 .8rem; color:#4B4A42; font-size:.86rem; }
    .workflow { display:flex; flex-wrap:wrap; gap:.45rem; align-items:center; color:#596E81; font-size:.82rem;
        margin:.55rem 0 1rem; }
    .workflow span { background:#EAF0F5; padding:.35rem .62rem; border-radius:999px; }
    .workflow b { color:#254F70; margin-right:.2rem; }
    .section-label { color:#687C8F; font-size:.72rem; font-weight:750; text-transform:uppercase; letter-spacing:.11em; margin-bottom:.15rem; }
    .control-panel { background:white; border:1px solid var(--line); border-radius:12px; padding:.9rem 1rem .35rem; }
    .comparison-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.7rem; margin:.45rem 0 1rem; }
    .compare-card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.9rem 1rem .85rem; min-width:0;
        box-shadow:0 1px 2px rgba(17,38,61,.025); }
    .compare-title { font-size:.8rem; font-weight:700; color:#536A7E; min-height:1.2rem; }
    .compare-values { display:grid; grid-template-columns:1fr 1fr; gap:.5rem; margin:.65rem 0 .7rem; }
    .compare-values span { display:block; color:#7A8A99; text-transform:uppercase; font-size:.62rem; letter-spacing:.075em; font-weight:700; margin-bottom:.13rem; }
    .compare-values strong { color:#6B7A88; font-size:1.19rem; font-weight:650; white-space:nowrap; }
    .compare-values .proposed-value strong { color:var(--navy); font-size:1.35rem; font-weight:750; }
    .change { display:inline-block; border-radius:999px; padding:.25rem .55rem; font-size:.7rem; font-weight:700; }
    .change.positive { color:#176545; background:#E8F4EE; }
    .change.negative { color:#9A3D3D; background:#FAECEB; }
    .change.neutral { color:#596E81; background:#EDF2F6; }
    .context-card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem; }
    .small-muted { color:#718395; font-size:.82rem; }
    .stTabs [data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { color:#597086; font-weight:650; padding:.55rem .85rem; }
    [data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:10px; padding:.7rem .85rem; }
    @media (max-width: 1050px) { .comparison-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width: 620px) {
      .block-container { padding-left:1rem; padding-right:1rem; }
      .hero { padding:1.1rem 1.1rem; }
      h1 { font-size:1.7rem !important; }
      .comparison-grid { grid-template-columns:1fr 1fr; gap:.45rem; }
      .compare-card { padding:.7rem .7rem; }
      .compare-values strong { font-size:1rem; }
      .compare-values .proposed-value strong { font-size:1.12rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="hero"><div class="eyebrow">Illustrative portfolio analytics · advisor workspace</div>'
    '<h1>Portfolio Impact Simulator</h1>'
    '<p>See how a hypothetical alternatives allocation may change a traditional portfolio’s return, risk and stress response.</p></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer"><b>Illustrative only.</b> Assumptions and outputs are hypothetical, not actual fund performance, forecasts or backtests. '
    'This tool is not investment advice. Diversification does not guarantee a profit or protect against loss.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="workflow"><span><b>1</b> Set portfolio mix</span><span><b>2</b> Compare modeled outcomes</span>'
    '<span><b>3</b> Explore stress scenarios</span><span><b>4</b> Download advisor brief</span></div>',
    unsafe_allow_html=True,
)

expected, vol, fees, corr = defaults()

with st.sidebar:
    st.markdown("### Model assumptions")
    st.caption("Editable annual assumptions. These examples do not represent an actual strategy or fund.")
    with st.expander("Capital-market assumptions", expanded=True):
        rf = st.number_input("Risk-free rate (%)", min_value=-5.0, max_value=20.0, value=3.0, step=0.1, format="%.1f") / 100
        assumptions = pd.DataFrame({"Expected return": expected, "Volatility": vol, "Annual fee": fees})
        edited = st.data_editor(
            assumptions,
            column_config={
                "Expected return": st.column_config.NumberColumn(format="percent", min_value=-1.0, max_value=1.0, step=0.0025),
                "Volatility": st.column_config.NumberColumn(format="percent", min_value=0.0, max_value=2.0, step=0.0025),
                "Annual fee": st.column_config.NumberColumn(format="percent", min_value=0.0, max_value=1.0, step=0.0005),
            },
            key="assumption_editor",
            **stretch_kwargs(st.data_editor),
        )
        expected, vol, fees = edited["Expected return"], edited["Volatility"], edited["Annual fee"]
        st.caption("Fees reduce expected return once. Constant management fees do not add modeled volatility.")
        st.markdown("**Asset-class correlations**")
        corr = st.data_editor(
            corr,
            key="correlation_editor",
            column_config={a: st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, format="%.2f", step=0.05) for a in ASSETS},
            **stretch_kwargs(st.data_editor),
        )
        st.caption("Matrix must be symmetric, have a unit diagonal and be positive semidefinite.")

    with st.expander("How the model works"):
        st.markdown(
            "Expected net return = weighted gross return less weighted annual fees. "
            "Portfolio volatility uses the asset covariance matrix. Sharpe uses net return less the risk-free rate. "
            "Stress results apply the displayed asset-return shocks to the selected weights and subtract one year of fees."
        )

with st.container():
    st.markdown('<div class="section-label">Step 1 · Portfolio design</div>', unsafe_allow_html=True)
    control_cols = st.columns([1.0, 1.0, 1.25])
    with control_cols[0]:
        st.markdown("**Starting portfolio**")
        base_equity = st.slider("Starting equities", 0, 100, 60, 1, format="%d%%") / 100
        st.caption(f"Fixed income: {pct(1 - base_equity, 0)} · default mix is 60/40")
    with control_cols[1]:
        st.markdown("**Proposed portfolio**")
        allocation_mode = st.radio(
            "Funding method",
            ["Fund proportionally", "Set custom weights"],
            help="Proportional funding preserves your starting equity/bond mix. In custom mode, fixed income is the remainder.",
        )
    with control_cols[2]:
        st.markdown("**Alternative allocation**")
        alt_weight = st.slider("Allocate to alternatives", 0, 30, 10, 1, format="%d%%") / 100
        st.caption("Funded from the traditional portfolio · 0–30%")

starting = pd.Series({"Equities": base_equity, "Fixed income": 1 - base_equity})
if allocation_mode == "Fund proportionally":
    proposed = proportional_alternative_allocation(starting.to_dict(), alt_weight)
else:
    custom_equity = st.slider(
        "Proposed equities",
        0,
        int(round(100 * (1 - alt_weight))),
        int(round(60 * (1 - alt_weight))),
        1,
        format="%d%%",
    ) / 100
    proposed = pd.Series({"Equities": custom_equity, "Fixed income": 1 - alt_weight - custom_equity, "Alternatives": alt_weight})
    validate_weights(proposed.to_dict())

try:
    validate_assumptions(expected.to_dict(), vol.to_dict(), fees.to_dict(), corr)
    before = portfolio_metrics(
        {"Equities": starting.Equities, "Fixed income": starting["Fixed income"], "Alternatives": 0.0},
        expected, vol, fees, corr, rf,
    )
    after = portfolio_metrics(proposed, expected, vol, fees, corr, rf)
except (PortfolioInputError, ValueError, TypeError) as exc:
    st.error(f"Review the assumptions or allocations: {exc}")
    st.stop()

st.markdown('<div class="section-label">Step 2 · Before-and-after comparison</div>', unsafe_allow_html=True)
st.markdown("### Portfolio impact at a glance")
st.caption("Annualized model estimates · proposed results update with your inputs")
delta_return = after.net_return - before.net_return
delta_vol = after.volatility - before.volatility
delta_sharpe = after.sharpe - before.sharpe
delta_fee = after.fee_drag - before.fee_drag
cards = [
    comparison_card("Net expected annual return", pct(before.net_return), pct(after.net_return), f"{delta_return:+.2%}", None if np.isclose(delta_return, 0.0, atol=1e-12) else delta_return > 0),
    comparison_card("Annualized volatility", pct(before.volatility), pct(after.volatility), f"{delta_vol:+.2%}", None if np.isclose(delta_vol, 0.0, atol=1e-12) else delta_vol < 0),
    comparison_card("Net Sharpe ratio", sharpe_text(before.sharpe), sharpe_text(after.sharpe), f"{delta_sharpe:+.2f}", None if np.isclose(delta_sharpe, 0.0, atol=1e-12) else delta_sharpe > 0),
    comparison_card("Annual fee drag", pct(before.fee_drag), pct(after.fee_drag), f"{delta_fee:+.2%}", None if np.isclose(delta_fee, 0.0, atol=1e-12) else delta_fee < 0),
]
st.markdown('<div class="comparison-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
st.caption("Green indicates a favorable change, red indicates deterioration, and gray indicates no change. A favorable change in one measure does not imply an overall improvement.")
brief_slot = st.empty()

tab_overview, tab_stress, tab_assumptions = st.tabs(["Portfolio comparison", "Scenario analysis", "Assumptions & method"])

with tab_overview:
    st.markdown("#### Allocation before and after")
    alloc = pd.DataFrame(
        {"Portfolio": ["Starting portfolio", "Proposed"] * 3, "Asset class": ASSETS * 2, "Weight": [0.0] * 6}
    )
    alloc.loc[alloc.Portfolio == "Starting portfolio", "Weight"] = [starting.Equities, starting["Fixed income"], 0.0]
    alloc.loc[alloc.Portfolio == "Proposed", "Weight"] = proposed.reindex(ASSETS).to_numpy()
    allocation_fig = px.bar(
        alloc, x="Portfolio", y="Weight", color="Asset class", barmode="stack",
        text=alloc.Weight.map(lambda x: f"{x:.0%}"),
        color_discrete_map={"Equities": BLUE, "Fixed income": TEAL, "Alternatives": GOLD},
        title="Portfolio allocation · starting vs proposed",
        hover_data={"Weight": ":.1%", "Portfolio": True, "Asset class": True},
    )
    allocation_fig.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1%}<extra></extra>")
    allocation_fig.update_layout(
        height=355, margin=dict(l=15, r=15, t=55, b=15),
        yaxis=dict(title="Portfolio weight", tickformat=".0%", range=[0, 1]), xaxis_title=None,
        legend=dict(orientation="h", y=-.2), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Aptos, sans-serif", color="#334A5F"),
    )
    st.plotly_chart(allocation_fig, **stretch_kwargs(st.plotly_chart))

    return_risk_cols = st.columns([1.2, .8])
    with return_risk_cols[0]:
        profile = pd.DataFrame({
            "Portfolio": ["Starting portfolio", "Proposed portfolio"],
            "Expected net return": [before.net_return, after.net_return],
            "Annualized volatility": [before.volatility, after.volatility],
            "Net Sharpe ratio": [before.sharpe, after.sharpe],
        })
        profile_fig = go.Figure()
        profile_fig.add_trace(go.Scatter(
            x=profile["Annualized volatility"], y=profile["Expected net return"],
            mode="markers+text", text=profile["Portfolio"], textposition="top center",
            marker=dict(size=17, color=[GRAY, BLUE], line=dict(color=NAVY, width=1.5)),
            customdata=profile[["Net Sharpe ratio"]].to_numpy(),
            hovertemplate="<b>%{text}</b><br>Expected net return: %{y:.2%}<br>Annualized volatility: %{x:.2%}<br>Net Sharpe: %{customdata[0]:.2f}<extra></extra>",
        ))
        profile_fig.add_hline(y=rf, line_dash="dot", line_color="#9AA9B6", annotation_text="Risk-free rate")
        profile_fig.update_layout(
            title="Modeled return and risk", height=340, margin=dict(l=15, r=15, t=55, b=15),
            xaxis_title="Annualized volatility", yaxis_title="Expected net return",
            xaxis_tickformat=".0%", yaxis_tickformat=".0%", plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter, Aptos, sans-serif", color="#334A5F"),
        )
        st.plotly_chart(profile_fig, **stretch_kwargs(st.plotly_chart))
    with return_risk_cols[1]:
        st.markdown("#### Portfolio statistics")
        mt = metric_table(before, after)
        formatted = mt.style.format("{:.2%}", subset=pd.IndexSlice[mt.index[:-1], :]).format("{:.2f}", subset=pd.IndexSlice["Sharpe ratio (net)", :])
        st.dataframe(formatted, hide_index=False, **stretch_kwargs(st.dataframe))
        st.caption("Sharpe ratio uses net expected return after the displayed annual fee drag.")

    st.markdown("#### Risk contribution by asset class")
    rc_data = pd.DataFrame({
        "Asset class": ASSETS * 2,
        "Portfolio": ["Starting portfolio"] * 3 + ["Proposed portfolio"] * 3,
        "Contribution": list(before.risk_contributions.reindex(ASSETS).values) + list(after.risk_contributions.reindex(ASSETS).values),
    })
    risk_fig = px.bar(
        rc_data, x="Asset class", y="Contribution", color="Portfolio", barmode="group",
        color_discrete_map={"Starting portfolio": GRAY, "Proposed portfolio": BLUE},
        title="Component contributions to annualized volatility",
        hover_data={"Contribution": ":.2%", "Portfolio": True},
    )
    risk_fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}<br>Contribution: %{y:.2%}<extra></extra>")
    risk_fig.update_layout(height=330, margin=dict(l=15, r=15, t=55, b=15), yaxis_title="Contribution to volatility", yaxis_tickformat=".0%", plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Inter, Aptos, sans-serif", color="#334A5F"))
    st.plotly_chart(risk_fig, **stretch_kwargs(st.plotly_chart))
    st.caption("Component contributions sum to total portfolio volatility. Negative contributions can occur when an asset offsets risk under the selected correlation assumptions.")

    correlation_fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=ASSETS, y=ASSETS, zmin=-1, zmax=1,
        colorscale=[[0, "#C77972"], [0.5, "#F5F7F9"], [1, BLUE]],
        text=corr.to_numpy(), texttemplate="%{text:.2f}",
        hovertemplate="%{y} and %{x}<br>Assumed correlation: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Correlation", tickformat=".1f"),
    ))
    correlation_fig.update_layout(title="Assumed asset-class correlations", height=330, margin=dict(l=15, r=15, t=55, b=15), paper_bgcolor="white", font=dict(family="Inter, Aptos, sans-serif", color="#334A5F"))
    st.plotly_chart(correlation_fig, **stretch_kwargs(st.plotly_chart))

    return_phrase = "higher" if delta_return > 0 else "lower" if delta_return < 0 else "unchanged"
    risk_phrase = "higher" if delta_vol > 0 else "lower" if delta_vol < 0 else "unchanged"
    fee_phrase = "higher" if delta_fee > 0 else "lower" if delta_fee < 0 else "unchanged"
    st.markdown("#### Advisor interpretation")
    def relative_sentence(measure: str, direction: str) -> str:
        if direction == "unchanged":
            return f"{measure} is unchanged relative to the starting portfolio"
        return f"{measure} is {direction} than the starting portfolio"

    st.markdown(
        f"<div class='context-card'>With these assumptions, {relative_sentence('expected net return', return_phrase)}; "
        f"{relative_sentence('modeled volatility', risk_phrase)}. Annual fee drag is {fee_phrase} relative to the starting portfolio. "
        "The modeled diversification effect comes from the selected correlations and can change materially when those assumptions change. "
        "Review return, risk, cost and stress outcomes together; no one measure determines whether an allocation is appropriate.</div>",
        unsafe_allow_html=True,
    )

with tab_stress:
    st.markdown("#### Illustrative stress scenarios")
    st.markdown(
        "Each scenario assigns a hypothetical return to each asset class. The portfolio result weights those returns by your selected allocation, "
        "then subtracts one year of modeled fees. These are simple illustrations, not forecasts or historical backtests."
    )
    scenario_defaults = pd.DataFrame(
        {
            "Equities": [-0.25, -0.10, -0.12, 0.0],
            "Fixed income": [0.03, -0.08, -0.04, 0.0],
            "Alternatives": [-0.12, -0.04, -0.02, 0.0],
        },
        index=["Equity selloff", "Rates rise", "Inflationary environment", "Custom shocks"],
    )
    with st.expander("Edit hypothetical asset-class shocks", expanded=False):
        edited_scenarios = st.data_editor(
            scenario_defaults,
            key="scenario_editor",
            column_config={a: st.column_config.NumberColumn(format="percent", min_value=-1.0, max_value=1.0, step=0.01) for a in ASSETS},
            **stretch_kwargs(st.data_editor),
        )
    scenario_rows = []
    try:
        for scenario_name, shock_row in edited_scenarios.iterrows():
            shock_map = shock_row.reindex(ASSETS).to_dict()
            starting_gross = scenario_impact(
                {"Equities": starting.Equities, "Fixed income": starting["Fixed income"], "Alternatives": 0.0}, shock_map
            )
            proposed_gross = scenario_impact(proposed, shock_map)
            starting_net = starting_gross - before.fee_drag
            proposed_net = proposed_gross - after.fee_drag
            scenario_rows.append({
                "Scenario": scenario_name,
                "Starting portfolio": starting_net,
                "Proposed portfolio": proposed_net,
                "Change vs. starting": proposed_net - starting_net,
            })
    except (PortfolioInputError, ValueError, TypeError) as exc:
        st.error(f"Review scenario shock inputs: {exc}")
        st.stop()
    scenario_results = pd.DataFrame(scenario_rows).set_index("Scenario")
    scenario_display = scenario_results.rename(columns={
        "Starting portfolio": "Starting net",
        "Proposed portfolio": "Proposed net",
        "Change vs. starting": "Difference",
    })
    styled_scenarios = scenario_display.style.format("{:.2%}").map(
        lambda value: f"color: {GREEN if value > 0 else RED if value < 0 else GRAY}; font-weight: 700",
        subset=["Difference"],
    )
    st.dataframe(styled_scenarios, **stretch_kwargs(st.dataframe))
    st.caption("Net scenario values deduct one year of weighted management fees. Difference is proposed minus starting.")

    scenario_names = list(scenario_results.index)
    scenario_diffs = scenario_results["Change vs. starting"].to_numpy()
    colors = [GREEN if value >= 0 else RED for value in scenario_diffs]
    scenario_fig = go.Figure()
    scenario_fig.add_trace(go.Bar(
        name="Starting portfolio", x=scenario_names, y=scenario_results["Starting portfolio"],
        marker_color=GRAY, text=scenario_results["Starting portfolio"].map(lambda x: f"{x:.1%}"),
        hovertemplate="<b>%{x}</b><br>Starting portfolio: %{y:.2%}<extra></extra>",
    ))
    scenario_fig.add_trace(go.Bar(
        name="Proposed portfolio", x=scenario_names, y=scenario_results["Proposed portfolio"],
        marker_color=colors, text=scenario_results["Proposed portfolio"].map(lambda x: f"{x:.1%}"),
        customdata=scenario_diffs,
        hovertemplate="<b>%{x}</b><br>Proposed portfolio: %{y:.2%}<br>Change vs. starting: %{customdata:+.2%}<extra></extra>",
    ))
    scenario_fig.add_hline(y=0, line_color="#758594", line_width=1)
    scenario_fig.update_layout(
        title="One-period portfolio impact · after one year of fees", barmode="group", height=390,
        margin=dict(l=15, r=15, t=60, b=20), xaxis_title=None,
        yaxis_title="Illustrative portfolio return", yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=1.13), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, Aptos, sans-serif", color="#334A5F"),
    )
    st.plotly_chart(scenario_fig, **stretch_kwargs(st.plotly_chart))
    st.caption("Green proposed bars outperform the starting portfolio in that scenario; red bars underperform. Net results subtract one year of weighted fees and exclude trading, taxes, liquidity, compounding and path dependence.")
    worst_name = scenario_results["Change vs. starting"].idxmin()
    best_name = scenario_results["Change vs. starting"].idxmax()
    st.info(
        f"Under the current shocks, the largest relative disadvantage is **{worst_name}** "
        f"({scenario_results.loc[worst_name, 'Change vs. starting']:+.2%} vs. starting), while the largest relative advantage is **{best_name}** "
        f"({scenario_results.loc[best_name, 'Change vs. starting']:+.2%})."
    )
    zero_scenarios = [name for name, row in edited_scenarios.iterrows() if np.isclose(row.to_numpy(dtype=float), 0).all()]
    if zero_scenarios:
        st.caption(f"{', '.join(zero_scenarios)} has no asset return shocks entered; its net result reflects one year of portfolio fees only.")

with tab_assumptions:
    st.markdown("#### Assumptions currently in use")
    detail = pd.DataFrame({
        "Expected annual gross return": expected,
        "Annualized volatility": vol,
        "Annual management fee": fees,
        "Starting weight": pd.Series({"Equities": starting.Equities, "Fixed income": starting["Fixed income"], "Alternatives": 0}),
        "Proposed weight": proposed,
    })
    st.dataframe(detail.style.format("{:.2%}"), **stretch_kwargs(st.dataframe))
    st.markdown("#### Calculation method")
    st.markdown(
        "- Expected gross return is the weighted sum of asset-class expected returns. Net return is gross return less weighted annual fees; fees are deducted once.\n"
        "- Covariance is volatilityᵢ × correlationᵢⱼ × volatilityⱼ. Portfolio volatility is √(wᵀΣw).\n"
        "- Net Sharpe is (expected net return − risk-free rate) ÷ annualized volatility.\n"
        "- Component risk contribution is wᵢ(Σw)ᵢ ÷ portfolio volatility. Contributions sum to total volatility when it is positive.\n"
        "- Scenario impact weights the specified one-period simple returns by the selected portfolio weights; displayed scenario outcomes subtract one year of weighted management fees."
    )
    st.markdown("#### Limitations")
    st.markdown(
        "This is a hypothetical mean-variance model. It does not model return distributions, changing correlations, leverage, illiquidity, "
        "valuation lags, taxes, rebalancing, withdrawals, compounding, manager selection or investment-specific terms. "
        "All assumptions are illustrative and do not represent actual fund performance."
    )

# Fill the top-of-page download slot with a brief built from the current scenarios and portfolio inputs.
brief = build_advisor_brief(
    {"Equities": starting.Equities, "Fixed income": starting["Fixed income"], "Alternatives": 0.0},
    proposed.to_dict(), before, after, expected.to_dict(), vol.to_dict(), fees.to_dict(), corr, rf,
    edited_scenarios.reindex(columns=ASSETS), scenario_results.reset_index().set_index("Scenario"),
)
with brief_slot.container():
    st.markdown("#### Step 4 · Advisor briefing")
    st.download_button(
        "Download advisor briefing",
        data=brief,
        file_name="advisor-portfolio-impact-brief.md",
        mime="text/markdown",
        type="primary",
    )
    st.caption("The downloaded brief uses the current portfolio, assumption and scenario inputs.")
