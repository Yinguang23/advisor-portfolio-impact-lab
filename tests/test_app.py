"""Streamlit-level smoke checks for live widgets and result refreshes."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_loads_and_updates_when_alternatives_slider_changes():
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception

    metrics = app.get("dataframe")[0].value
    assert metrics.loc["Expected net return", "Proposed portfolio"] == pytest.approx(0.05423)
    assert metrics.loc["Annualized volatility", "Proposed portfolio"] == pytest.approx(0.09081074826252673)

    app.get("slider")[1].set_value(0)
    app.run()
    assert not app.exception
    zero_alt_metrics = app.get("dataframe")[0].value
    assert zero_alt_metrics.loc["Expected net return", "Proposed portfolio"] == pytest.approx(
        zero_alt_metrics.loc["Expected net return", "Starting portfolio"]
    )
    assert zero_alt_metrics.loc["Annualized volatility", "Proposed portfolio"] == pytest.approx(
        zero_alt_metrics.loc["Annualized volatility", "Starting portfolio"]
    )
    rendered_markdown = "\n".join(str(element.value) for element in app.get("markdown"))
    assert "Unchanged" in rendered_markdown
    assert "has a unchanged" not in rendered_markdown

    app.get("slider")[1].set_value(20)
    app.run()
    assert not app.exception
    refreshed_metrics = app.get("dataframe")[0].value
    assert refreshed_metrics.loc["Expected gross return", "Proposed portfolio"] == pytest.approx(0.0578)
    assert refreshed_metrics.loc["Annual fee drag", "Proposed portfolio"] == pytest.approx(0.00404)
    assert refreshed_metrics.loc["Expected net return", "Proposed portfolio"] == pytest.approx(0.05376)
    assert refreshed_metrics.loc["Annualized volatility", "Proposed portfolio"] == pytest.approx(0.0866928371)

    for alternative_percent in (0, 10, 20, 30):
        app.get("slider")[1].set_value(alternative_percent)
        app.run()
        allocation = app.get("dataframe")[3].value["Proposed weight"]
        assert allocation.sum() == pytest.approx(1.0)
        assert allocation["Alternatives"] == pytest.approx(alternative_percent / 100)


def test_custom_weight_mode_loads_and_updates_without_stale_errors():
    app = AppTest.from_file(APP_PATH).run()
    app.get("radio")[0].set_value("Set custom weights")
    app.get("slider")[1].set_value(20)
    app.run()
    assert not app.exception
    proposed_weights = app.get("dataframe")[3].value["Proposed weight"]
    assert proposed_weights.sum() == pytest.approx(1.0)
    # The additional custom equity slider is bounded by the remaining weight,
    # so equities + alternatives + residual fixed income still equal 100%.
    assert app.get("slider")[2].value <= 80
    assert np.isfinite(app.get("dataframe")[0].value.to_numpy(dtype=float)).all()
