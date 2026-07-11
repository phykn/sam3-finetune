import torch

from src.finetune.loss import (
    class_weights,
    finetune_loss,
    mask_bce,
    mask_dice,
    mean_loss,
    target_iou,
)


def make_batch(mask_valid=(1.0, 0.0)):
    return {
        "target": torch.tensor(
            [
                [[[1.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ]
        ),
        "mask_valid": torch.tensor(mask_valid),
        "is_auto_bg": torch.tensor([0.0, 0.0]),
        "label_target": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "label_weight": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
    }


def make_output():
    return {
        "mask_logits": torch.zeros(2, 1, 2, 2, requires_grad=True),
        "iou_scores": torch.zeros(2, 1, requires_grad=True),
        "class_logits": torch.zeros(2, 1, 2, requires_grad=True),
    }


def test_mask_bce_keeps_gradient_for_confident_wrong_logit():
    logits = torch.tensor([[[[-10.0]]]], requires_grad=True)

    loss = mask_bce(logits, torch.ones_like(logits)).sum()
    loss.backward()

    assert logits.grad.item() < -0.9


def test_mask_dice_matches_soft_overlap_equation():
    logits = torch.zeros(1, 1, 1, 2)
    target = torch.tensor([[[[1.0, 0.0]]]])

    loss = mask_dice(logits, target)

    expected = 1 - (2 * 0.5 + 1) / (0.5 + 0.5 + 1 + 1)
    assert torch.allclose(loss, torch.tensor([expected]))


def test_target_iou_uses_binary_mask_thresholds():
    logits = torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]])
    target = torch.tensor([[[[1.0, 0.6], [0.4, 0.0]]]])

    iou = target_iou(logits, target)

    assert torch.allclose(iou, torch.tensor([[1 / 3]]))


def test_auto_background_uses_detached_particle_probability():
    logits = torch.tensor([[2.0, -1.0], [-2.0, 1.0]], requires_grad=True)
    weights = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    auto = torch.tensor([True, False])

    adjusted = class_weights(weights, logits, auto)

    assert torch.allclose(adjusted[0, 0], 1 - logits[0, 0].sigmoid())
    assert adjusted[1, 0] == 1
    assert adjusted.requires_grad is False


def test_auto_background_weight_reduces_class_gradient():
    particle = 0.9
    logit = torch.logit(torch.tensor(particle))
    batch = {
        "target": torch.zeros(1, 1, 1, 1),
        "mask_valid": torch.zeros(1),
        "is_auto_bg": torch.ones(1),
        "label_target": torch.zeros(1, 1),
        "label_weight": torch.ones(1, 1),
    }
    out = {
        "mask_logits": torch.zeros(1, 1, 1, 1, requires_grad=True),
        "iou_scores": torch.zeros(1, 1, requires_grad=True),
        "class_logits": logit.reshape(1, 1, 1).requires_grad_(),
    }

    loss, _stats = finetune_loss(batch, out)
    loss.backward()

    expected = particle * (1 - particle)
    assert torch.allclose(out["class_logits"].grad, torch.tensor([[[expected]]]))


def test_background_samples_do_not_contribute_mask_or_iou_loss():
    batch = make_batch()
    out = make_output()
    changed = make_output()
    with torch.no_grad():
        changed["mask_logits"][1].fill_(100)
        changed["iou_scores"][1].fill_(100)

    total, stats = finetune_loss(batch, out)
    changed_total, changed_stats = finetune_loss(batch, changed)

    assert torch.allclose(total, changed_total)
    assert stats == changed_stats


def test_iou_loss_compares_sigmoid_score_with_target_iou():
    batch = make_batch()
    out = make_output()
    with torch.no_grad():
        out["mask_logits"][0, 0] = torch.tensor([[10.0, 10.0], [-10.0, -10.0]])
        out["iou_scores"][0, 0] = 10.0

    _total, stats = finetune_loss(batch, out)

    assert stats["iou_loss"] < 1e-6


def test_no_valid_masks_keeps_class_loss_trainable():
    batch = make_batch(mask_valid=(0.0, 0.0))
    out = make_output()

    total, stats = finetune_loss(batch, out)
    total.backward()

    assert stats["mask_bce"] == 0.0
    assert stats["mask_dice"] == 0.0
    assert stats["iou_loss"] == 0.0
    assert out["class_logits"].grad is not None
    assert torch.isfinite(out["class_logits"].grad).all()


def test_class_stats_report_each_label_and_ignore_auto_background_accuracy():
    batch = make_batch()
    batch["is_auto_bg"] = torch.tensor([0.0, 1.0])
    out = make_output()
    out["class_logits"] = torch.tensor(
        [[[2.0, -2.0]], [[2.0, 2.0]]],
        requires_grad=True,
    )

    _loss, stats = finetune_loss(batch, out)

    assert stats["class_acc_0"] == 1.0
    assert stats["class_acc_1"] == 1.0
    assert stats["class_loss_0"] > 0
    assert stats["class_loss_1"] > 0


def test_mean_loss_scales_backward_for_ddp_gradient_average(monkeypatch):
    reduced = iter((torch.tensor(4.0), torch.tensor(10.0)))
    monkeypatch.setattr("src.finetune.loss.world_size", lambda: 2)
    monkeypatch.setattr("src.finetune.loss.sum_value", lambda _value: next(reduced))
    local_sum = torch.tensor(3.0, requires_grad=True)

    loss, logged = mean_loss(local_sum, torch.tensor(2.0))
    loss.backward()

    assert loss.item() == 1.5
    assert local_sum.grad.item() == 0.5
    assert logged == 2.5
