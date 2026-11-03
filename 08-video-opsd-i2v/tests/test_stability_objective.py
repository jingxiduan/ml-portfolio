import torch

from src.opsd_trainer_i2v import compute_fit_loss
from src.rewards.naturalness_penalty import NaturalnessPenalty


def test_anchored_objective_is_bounded_and_penalizes_drift():
    anchor = torch.zeros(1, 1, 1, 1, 1)
    plus = torch.ones_like(anchor)
    minus = -torch.ones_like(anchor)

    at_anchor = compute_fit_loss(anchor, plus, minus, anchor, "anchored", 0.25)[0]
    far = compute_fit_loss(torch.full_like(anchor, 10.0), plus, minus, anchor, "anchored", 0.25)[0]

    assert at_anchor.item() == 1.0
    assert far.item() > at_anchor.item()


def test_contrastive_mode_matches_historical_formula():
    pred = torch.tensor([0.5])
    plus = torch.tensor([1.0])
    minus = torch.tensor([-1.0])
    anchor = torch.tensor([0.0])

    loss, loss_plus, loss_minus, _ = compute_fit_loss(
        pred, plus, minus, anchor, "contrastive", 0.0
    )

    assert torch.allclose(loss, loss_plus - loss_minus)


def test_tanh_pixel_mapping_keeps_extreme_gradient():
    video = torch.full((1, 2, 3, 8, 8), 1.5, requires_grad=True)
    reward = NaturalnessPenalty(input_mapping="tanh")(video, ["test"])
    gradient = torch.autograd.grad(reward.sum(), video)[0]

    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0
