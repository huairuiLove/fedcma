from types import SimpleNamespace

import numpy as np
import pytest
import torch

from curriculum import curriculum_plan, training_order


def cfg(**kw):
    return SimpleNamespace(**{"curriculum_schedule": "warmup_bounded", "warmup_rounds": 1,
                              "rounds": 5, "rho0": .55, "xi": .7, **kw})


def test_warmup_is_full_data_then_threshold_covers_reachable_scores():
    plans = [curriculum_plan(cfg(), t, True, .5) for t in range(5)]
    assert plans[0]["phase"] == "warmup" and not plans[0]["active"]
    np.testing.assert_allclose([p["rho"] for p in plans[1:]], [.55, 7/12, 37/60, .65])
    assert all(p["active"] for p in plans[1:])
    hardest = torch.tensor(.7) * .5 + .3
    assert hardest <= plans[-1]["rho"] + plans[-1]["tolerance"]
    assert all(not curriculum_plan(cfg(), t, False, .5)["active"] for t in range(5))


def test_legacy_schedule_and_training_seed_order_are_preserved():
    legacy = SimpleNamespace(rounds=5, rho0=.6)
    np.testing.assert_allclose([curriculum_plan(legacy, t, True, .5)["rho"] for t in range(5)],
                               [.6, .7, .8, .9, 1.])
    selected = torch.arange(73)
    a, b = torch.Generator().manual_seed(91), torch.Generator().manual_seed(91)
    expected = selected[torch.randperm(len(selected), generator=a)]
    torch.testing.assert_close(training_order(selected, 73, b), expected, rtol=0, atol=0)


@pytest.mark.parametrize("target", [3, 8, 64, 83])
def test_repetition_hits_exact_budget_with_balanced_multiplicities(target):
    selected = torch.tensor([2, 7, 19])
    a = training_order(selected, target, torch.Generator().manual_seed(7))
    b = training_order(selected, target, torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(a) == target
    unique, counts = a.unique(return_counts=True)
    torch.testing.assert_close(unique, selected)
    assert counts.max() - counts.min() <= 1
    assert sum(len(part) for part in a.split(64)) == target


def test_invalid_schedule_and_empty_pool_fail_explicitly():
    with pytest.raises(ValueError, match="warmup_rounds"):
        curriculum_plan(cfg(warmup_rounds=5), 0, True, .5)
    with pytest.raises(ValueError, match="score upper bound"):
        curriculum_plan(cfg(rho0=.7), 1, True, .5)
    with pytest.raises(ValueError, match="nonempty"):
        training_order(torch.tensor([], dtype=torch.long), 8, torch.Generator())
