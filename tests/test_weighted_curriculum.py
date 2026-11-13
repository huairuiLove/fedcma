from types import SimpleNamespace

import pytest
import torch

from curriculum import balanced_hardness_weights, curriculum_plan, joint_groups
from model import FedCMAModel


def test_weight_schedule_warms_up_then_gradually_increases():
    cfg = SimpleNamespace(curriculum_schedule="balanced_weight", rounds=5, warmup_rounds=2, curriculum_weight_max=.5)
    plans = [curriculum_plan(cfg, t, True, .5) for t in range(5)]
    assert [p["weight_strength"] for p in plans] == pytest.approx([0, 0, 1/6, 1/3, .5])
    assert [p["active"] for p in plans] == [False, False, True, True, True]
    assert not curriculum_plan(cfg, 3, False, .5)["active"]
    cfg.curriculum_weight_max = 1.
    with pytest.raises(ValueError, match="curriculum_weight_max"):
        curriculum_plan(cfg, 3, True, .5)


def test_weights_treat_ties_and_singletons_without_group_mass_shift():
    losses = torch.tensor([.1, .5, .5, .9, 3.], requires_grad=True)
    labels = torch.tensor([0., 0., 0., 0., 1.])
    masks = torch.ones(5, 2)
    weights = balanced_hardness_weights(losses, labels, masks, .5)
    torch.testing.assert_close(weights, torch.tensor([.5, 1., 1., 1.5, 1.]), rtol=0, atol=0)
    assert not weights.requires_grad
    torch.testing.assert_close(balanced_hardness_weights(losses, labels, masks, 0.), torch.ones(5))


@pytest.mark.parametrize("strength", [.25, .5, .75])
def test_weight_bounds_group_sums_and_permutation_invariance(strength):
    generator = torch.Generator().manual_seed(43)
    losses = torch.rand(64, generator=generator)
    labels = (torch.arange(64) % 3 == 0).float()
    masks = torch.tensor([[1., 1.], [0., 1.], [1., 0.], [1., 1.]])[torch.arange(64) % 4]
    weights = balanced_hardness_weights(losses, labels, masks, strength)
    groups = joint_groups(labels, masks)
    for group in groups.unique():
        which = groups == group
        assert weights[which].sum().item() == pytest.approx(int(which.sum()), abs=2e-6)
    assert weights.min() >= 1 - strength and weights.max() <= 1 + strength
    order = torch.randperm(64, generator=generator)
    changed = balanced_hardness_weights(losses[order], labels[order], masks[order], strength)
    torch.testing.assert_close(changed, weights[order], rtol=0, atol=0)


def test_weighted_task_gradient_matches_per_sample_importance():
    cfg = SimpleNamespace(method="fedavg", embed_dim=8, hidden_dim=4, prior_hidden=4,
                          num_heads=2, dropout=0., lambda_align=.001, lambda_reg=.001)
    model = FedCMAModel(cfg)
    logits = torch.tensor([-2., 0., 1., 3.], requires_grad=True)
    labels = torch.tensor([0., 1., 0., 1.])
    weights = torch.tensor([.5, 1.5, 1.5, .5])
    loss, parts = model.loss(logits, {"label": labels}, None, sample_weights=weights)
    loss.backward()
    torch.testing.assert_close(logits.grad, weights * (logits.detach().sigmoid() - labels) / len(labels))
    expected_raw = torch.nn.functional.binary_cross_entropy_with_logits(logits.detach(), labels)
    torch.testing.assert_close(parts["task_unweighted"], expected_raw)
    assert parts["align"] == 0 and parts["reg"] == 0
