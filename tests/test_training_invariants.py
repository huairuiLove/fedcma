from types import SimpleNamespace
import numpy as np
import pytest
import torch

from dataset import f_impute, make_masks, partition_patients, fit_normalizer
from fed import aggregate, difficulty_score, proximal_penalty
from model import FedCMAModel, ModalityFusion
from methods import METHODS, method_components


def config(method="fedcma"):
    return SimpleNamespace(method=method, embed_dim=128, hidden_dim=16, prior_hidden=64,
                           num_heads=4, dropout=0., align_detach=False,
                           lambda_align=.001, lambda_reg=.001)


def batch():
    return {"ts": torch.randn(4, 48, 34), "summary": torch.randn(4, 102),
            "text": torch.randn(4, 768), "masks": torch.tensor([[1., 0.]] * 4),
            "label": torch.tensor([0., 1., 0., 1.])}


def test_forward_fill_gaps_and_same_hour_observations():
    values = np.array([[2., 0.], [0., 3.], [4., 0.]], np.float32)
    masks = np.array([[1, 0], [0, 1], [1, 0]], np.float32)
    grid = f_impute(values, masks, np.array([0.1, 0.2, 3.]), window=5)
    np.testing.assert_array_equal(grid[:, :2], [[2, 3], [2, 3], [2, 3], [4, 3], [4, 3]])
    np.testing.assert_array_equal(grid[:, 2:], [[1, 1], [0, 0], [0, 0], [1, 0], [0, 0]])


def test_missing_rate_and_determinism():
    a = make_masks(101, .5, np.random.default_rng(42))
    b = make_masks(101, .5, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert (a.sum(1) == 1).sum() == round(101 * .5)
    assert a.sum(1).min() >= 1
    with pytest.raises(ValueError):
        make_masks(10, 1.1, np.random.default_rng(1))


def test_partition_is_patient_grouped_and_reproducible():
    ids = np.repeat(np.arange(100).astype(str), 2)
    labels = np.repeat(np.arange(100) % 2, 2)
    a = partition_patients(ids, labels, 5, .5, np.random.default_rng(42))
    b = partition_patients(ids, labels, 5, .5, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert all(len(set(a[ids == patient])) == 1 for patient in np.unique(ids))
    assert np.bincount(a).min() > 0


def test_normalizer_ignores_unobserved_values():
    x = np.ones((10, 17), np.float32)
    mask = np.ones_like(x)
    mask[0] = 0
    a = fit_normalizer([{"irg_ts": x.copy(), "irg_ts_mask": mask}])
    x[0] = 1e20
    b = fit_normalizer([{"irg_ts": x, "irg_ts_mask": mask}])
    assert a == b


@pytest.mark.parametrize("method", list(METHODS))
def test_missing_modality_content_cannot_change_prediction(method):
    torch.manual_seed(3)
    model = FedCMAModel(config(method)).eval()
    x = batch()
    priors = torch.randn(2, 128) if method_components(method)["prior"] else None
    original, _ = model(x, priors)
    x["text"] = torch.randn_like(x["text"]) * 1000
    changed, _ = model(x, priors)
    torch.testing.assert_close(original, changed, rtol=0, atol=0)


def test_modality_attention_has_two_competing_keys_and_qk_gradients():
    torch.manual_seed(4)
    fusion = ModalityFusion(128, 4, 0.)
    output, weights, pool = fusion(torch.randn(3, 2, 128), return_weights=True)
    assert weights.shape[-2:] == (2, 2)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    assert weights.std() > .01
    output[:, 0].sum().backward()
    assert fusion.qkv.weight.grad[:256].abs().sum() > 0


def test_alignment_uses_only_complete_samples_and_trains_prior():
    model = FedCMAModel(config())
    x = batch()
    p = torch.randn(2, 128, requires_grad=True)
    logits, real = model(x, p)
    _, parts = model.loss(logits, x, real, p)
    assert parts["align"].item() == 0
    x["masks"][:] = 1
    logits, real = model(x, p)
    loss, parts = model.loss(logits, x, real, p)
    expected = (real - model.generated(p).unsqueeze(0)).square().sum(-1).mean()
    torch.testing.assert_close(parts["align"], expected)
    loss.backward()
    assert p.grad.abs().sum() > 0


def test_patent_difficulty_and_original_sample_count_aggregation():
    scores = difficulty_score(torch.tensor([1., 2.]), torch.tensor([[1., 1.], [1., 0.]]), .5)
    torch.testing.assert_close(scores, torch.tensor([.25, .75]))
    avg = aggregate([{"x": torch.tensor([1.])}, {"x": torch.tensor([5.])}], [3, 1])
    torch.testing.assert_close(avg["x"], torch.tensor([2.]))


def test_baseline_initialization_and_fedprox_anchor():
    torch.manual_seed(42)
    avg = FedCMAModel(config("fedavg"))
    torch.manual_seed(42)
    ours = FedCMAModel(config("fedcma"))
    for name, value in avg.state_dict().items():
        torch.testing.assert_close(value, ours.state_dict()[name], rtol=0, atol=0)
    anchor = {n: p.detach().clone() for n, p in avg.named_parameters() if p.requires_grad}
    assert proximal_penalty(avg, anchor).item() == 0
    with torch.no_grad():
        next(avg.parameters()).add_(1)
    assert proximal_penalty(avg, anchor).item() > 0


@pytest.mark.parametrize("method,prior,curriculum", [
    ("fedcma", True, True), ("fedcma_wo_prior", False, True),
    ("fedcma_wo_cl", True, False), ("fedcma_wo_both", False, False),
])
def test_independent_ablation_components_and_generator_trainability(method, prior, curriculum):
    assert method_components(method) == {"prior": prior, "curriculum": curriculum, "proximal": False}
    model = FedCMAModel(config(method))
    assert all(p.requires_grad == prior for p in model.generators.parameters())


def test_removing_both_components_is_fedavg():
    assert method_components("fedcma_wo_both") == method_components("fedavg")
    torch.manual_seed(42)
    base = FedCMAModel(config("fedavg")).eval()
    torch.manual_seed(42)
    ablated = FedCMAModel(config("fedcma_wo_both")).eval()
    x = batch()
    base_logits, real = base(x)
    ablated_logits, ablated_real = ablated(x)
    torch.testing.assert_close(base_logits, ablated_logits, rtol=0, atol=0)
    base_loss, _ = base.loss(base_logits, x, real)
    ablated_loss, parts = ablated.loss(ablated_logits, x, ablated_real)
    torch.testing.assert_close(base_loss, ablated_loss, rtol=0, atol=0)
    assert parts["align"] == 0 and parts["reg"] == 0
