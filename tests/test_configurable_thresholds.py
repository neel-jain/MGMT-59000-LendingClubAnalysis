"""
test_configurable_thresholds.py
=================================
Unit tests for src/configurable_thresholds.py.

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.configurable_thresholds import (  # noqa: E402
    LendingActionRule, RiskThresholdConfig, RiskTierBoundary,
    load_threshold_config,
)


@pytest.fixture
def default_config():
    return RiskThresholdConfig()


def test_get_tier_boundaries(default_config):
    assert default_config.get_tier(0.0) == "Low Risk"
    assert default_config.get_tier(0.10) == "Low Risk"
    assert default_config.get_tier(0.15) == "Moderate Risk"  # boundary belongs to upper tier
    assert default_config.get_tier(0.34) == "Moderate Risk"
    assert default_config.get_tier(0.35) == "High Risk"
    assert default_config.get_tier(0.60) == "Very High Risk"
    assert default_config.get_tier(1.0) == "Very High Risk"


def test_get_action_matches_tier(default_config):
    assert default_config.get_action("Low Risk") == "Approve"
    assert default_config.get_action("Very High Risk") == "Decline"


def test_get_action_unknown_tier_raises(default_config):
    with pytest.raises(ValueError):
        default_config.get_action("Nonexistent Tier")


def test_get_rate_adjustment_bps(default_config):
    assert default_config.get_rate_adjustment_bps("Low Risk") < 0
    assert default_config.get_rate_adjustment_bps("Very High Risk") > default_config.get_rate_adjustment_bps("High Risk")


def test_get_loan_grade_monotonic_with_risk(default_config):
    low_risk_grade = default_config.get_loan_grade(0.02)
    high_risk_grade = default_config.get_loan_grade(0.9)
    grade_order = [b.grade for b in default_config.loan_grade_bands]
    assert grade_order.index(low_risk_grade) < grade_order.index(high_risk_grade)


def test_to_dict_and_from_dict_roundtrip(default_config):
    data = default_config.to_dict()
    restored = RiskThresholdConfig.from_dict(data)
    assert restored.get_tier(0.42) == default_config.get_tier(0.42)
    assert restored.get_action("High Risk") == default_config.get_action("High Risk")
    assert restored.loan_grade_bands[0].grade == default_config.loan_grade_bands[0].grade


def test_save_and_load_roundtrip(tmp_path, default_config):
    path = tmp_path / "thresholds.json"
    default_config.save(path)
    assert path.exists()

    loaded = RiskThresholdConfig.load(path)
    assert loaded.get_tier(0.5) == default_config.get_tier(0.5)


def test_load_bootstraps_defaults_when_missing(tmp_path):
    path = tmp_path / "does_not_exist_yet.json"
    assert not path.exists()
    config_obj = RiskThresholdConfig.load(path)
    assert path.exists()  # bootstrapped to disk
    assert config_obj.get_tier(0.05) == "Low Risk"


def test_validate_passes_for_default_config(default_config):
    default_config.validate()  # should not raise


def test_validate_detects_gap():
    bad_config = RiskThresholdConfig(
        risk_tiers=[
            RiskTierBoundary("Low Risk", 0.0, 0.3, "d"),
            RiskTierBoundary("High Risk", 0.5, 1.01, "d"),  # gap between 0.3 and 0.5
        ],
        lending_actions=[
            LendingActionRule("Low Risk", "Approve", "d"),
            LendingActionRule("High Risk", "Decline", "d"),
        ],
        interest_rate_adjustment_bps={"Low Risk": 0.0, "High Risk": 100.0},
    )
    with pytest.raises(ValueError):
        bad_config.validate()


def test_validate_detects_missing_action_rule():
    bad_config = RiskThresholdConfig(
        risk_tiers=[
            RiskTierBoundary("Low Risk", 0.0, 0.5, "d"),
            RiskTierBoundary("High Risk", 0.5, 1.01, "d"),
        ],
        lending_actions=[
            LendingActionRule("Low Risk", "Approve", "d"),
            # Missing "High Risk" action rule.
        ],
        interest_rate_adjustment_bps={"Low Risk": 0.0, "High Risk": 100.0},
    )
    with pytest.raises(ValueError):
        bad_config.validate()


def test_load_threshold_config_convenience_function(tmp_path):
    path = tmp_path / "thresholds.json"
    result = load_threshold_config(path)
    assert isinstance(result, RiskThresholdConfig)
    assert path.exists()
