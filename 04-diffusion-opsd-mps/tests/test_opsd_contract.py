from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opsd_train import dual_branch_loss, trust_region_targets


def test_trust_region_targets_are_symmetric_detached_and_bounded():
    torch.manual_seed(0)
    anchor = torch.randn(2, 4, 8, 8)
    gradient = torch.randn_like(anchor)
    plus, minus, ratio = trust_region_targets(anchor, gradient, rho=0.05)
    torch.testing.assert_close(plus - anchor, anchor - minus)
    torch.testing.assert_close(ratio, torch.full_like(ratio, 0.05), rtol=1e-5, atol=1e-6)
    assert not plus.requires_grad
    assert not minus.requires_grad


def test_dual_branch_loss_is_zero_for_exact_symmetric_solution():
    torch.manual_seed(1)
    z = torch.randn(1, 4, 8, 8)
    sigma = torch.tensor(0.5).view(1, 1, 1, 1)
    old = torch.randn_like(z)
    theta = old + 0.1 * torch.randn_like(z)
    target_plus = z - sigma * theta
    target_minus = z - sigma * (2 * old - theta)
    loss, positive, negative = dual_branch_loss(
        z, sigma, theta, old, target_plus, target_minus, beta=1.0
    )
    assert loss.item() < 1e-8
    assert positive.item() < 1e-8
    assert negative.item() < 1e-8
