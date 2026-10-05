import pytest

from nextcheck.replay.environment import ReplayEnvironment


PANELS = [{"id": "initial", "cost": 0, "features": ["f0"]}] + [
    {"id": f"p{index}", "cost": 1, "features": [f"f{index}"]} for index in range(1, 6)
]


def test_replay_reveals_only_charged_groups_once_and_bounds_purchases():
    replay = ReplayEnvironment(list(range(6)), 2, PANELS)
    observed = replay.initial_observation()
    assert observed.values == {"f0": 0.0}
    with pytest.raises(ValueError):
        replay.completed_label(completed=False)
    for index in range(1, 6):
        observed, charge = replay.reveal(observed, f"p{index}", remaining_budget=1)
        assert charge == 1
        assert set(observed.values) == {f"f{n}" for n in range(index + 1)}
    same, duplicate_charge = replay.reveal(observed, "p5", remaining_budget=0)
    assert same == observed
    assert duplicate_charge == 0
    assert len(observed.observed_groups) - 1 == 5
    assert replay.completed_label(completed=True) == 2


def test_replay_rejects_overspend_without_reveal():
    replay = ReplayEnvironment(list(range(6)), 2, PANELS)
    before = replay.initial_observation()
    with pytest.raises(ValueError):
        replay.reveal(before, "p1", remaining_budget=0)
    assert before.values == {"f0": 0.0}
