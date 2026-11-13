from types import SimpleNamespace

import pytest
import torch

from curriculum import curriculum_plan, joint_groups, stratified_curriculum_order
from fed import curriculum_selection


def test_ordering_schedule_has_warmup_and_two_full_recovery_rounds():
    cfg = SimpleNamespace(curriculum_schedule="stratified_order", warmup_rounds=1,
                          full_data_rounds=2, rounds=5, curriculum_strength=.5, xi=.7)
    plans = [curriculum_plan(cfg, t, True, .5) for t in range(5)]
    assert [r["active"] for r in plans] == [False, True, True, False, False]
    assert [r["order_strength"] for r in plans] == [0., .5, .25, 0., 0.]
    assert all(r["coverage"] == "every_sample_once_per_epoch" for r in plans)
    cfg.full_data_rounds = 4
    with pytest.raises(ValueError, match="active round"):
        curriculum_plan(cfg, 0, True, .5)


@pytest.mark.parametrize("strength", [0., .25, .5, 1.])
def test_ordering_preserves_all_samples_and_every_batch_label_mask_counts(strength):
    labels = torch.tensor([0., 1., 0., 1.] * 9)
    masks = torch.tensor([[1., 1.], [0., 1.], [1., 0.]] * 12)
    local = torch.arange(len(labels))
    base = local[torch.randperm(len(local), generator=torch.Generator().manual_seed(7))]
    scores = torch.linspace(0., 1., len(local))
    result = stratified_curriculum_order(base, local, scores, labels, masks, strength)
    torch.testing.assert_close(result.sort().values, local, rtol=0, atol=0)
    torch.testing.assert_close(joint_groups(labels, masks)[result], joint_groups(labels, masks)[base], rtol=0, atol=0)
    for new, old in zip(result.split(7), base.split(7)):
        torch.testing.assert_close(labels[new], labels[old], rtol=0, atol=0)
        torch.testing.assert_close(masks[new], masks[old], rtol=0, atol=0)
    if strength == 0:
        torch.testing.assert_close(result, base, rtol=0, atol=0)
    if strength == 1:
        groups = joint_groups(labels, masks)[result]
        for group in groups.unique():
            difficulty = scores[result[groups == group]]
            assert (difficulty.diff() >= 0).all()
        assert not torch.equal(result, base)


def test_client_difficulty_is_independent_of_scoring_chunk_boundaries():
    class FixedLogits(torch.nn.Module):
        def forward(self, batch, prior):
            return batch["logits"], None

    data = {"logits": torch.tensor([-4., 0., 1., 2., -2., 6.]),
            "label": torch.tensor([0., 0., 1., 1., 1., 0.]),
            "masks": torch.tensor([[1., 1.], [0., 1.], [1., 0.]] * 2)}
    cfg = SimpleNamespace(batch_size=2, xi=.7, curriculum_schedule="stratified_order")
    indices = torch.arange(6)
    kept, first = curriculum_selection(FixedLogits(), data, indices, None, cfg, .65)
    cfg.batch_size = 3
    _, second = curriculum_selection(FixedLogits(), data, indices, None, cfg, .65)
    torch.testing.assert_close(kept, indices, rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
