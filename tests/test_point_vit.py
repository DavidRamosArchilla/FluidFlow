"""PointViT must ignore the padding added by drivaernet_collate_fn."""
import pytest
import torch

from data.load_drivaernet import drivaernet_collate_fn
from fluidFlow.point_vit import PointViT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _model():
    torch.manual_seed(0)
    model = PointViT(patch_size=16, hidden_size=256, depth=3, num_heads=4, num_frequencies=6).cuda()
    torch.nn.init.normal_(model.final_layer.weight, std=0.02)  # non-zero output (zero-initialised otherwise)
    return model


def _car(n):
    return torch.randn(1, n), torch.randn(3, n)


def test_prediction_does_not_depend_on_padding():
    model = _model()
    short, long_ = _car(30005), _car(41000)
    alone = {k: v.cuda() for k, v in drivaernet_collate_fn([short], pad_multiple=16).items()}
    batched = {k: v.cuda() for k, v in drivaernet_collate_fn([long_, short], pad_multiple=16).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_alone = model.sample(alone["context"], alone["mask"])
        pred_batched = model.sample(batched["context"], batched["mask"])
    torch.testing.assert_close(pred_alone[0, :, :30005], pred_batched[1, :, :30005], atol=2e-2, rtol=2e-2)
    assert (pred_batched[1, :, 30005:] == 0).all()


def test_loss_ignores_padded_targets():
    model = _model()
    batch = {k: v.cuda() for k, v in drivaernet_collate_fn([_car(41000), _car(30005)], pad_multiple=16).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch["x"], batch["context"], batch["mask"])
        batch["x"][1, :, 30005:] = float("nan")
        loss_nan_padding = model(batch["x"], batch["context"], batch["mask"])
    assert torch.isfinite(loss_nan_padding)
    torch.testing.assert_close(loss, loss_nan_padding)
