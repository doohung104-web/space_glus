"""Phase-2A inference MVP self-checks.

Pure torch + numpy.  Verifies the four scenarios from §9:

    A. use_trajectory=False              -> SEG embedding unchanged
    B. use_trajectory=True, empty hist  -> trajectory inputs are None
    C. use_trajectory=True, non-empty   -> modulation fires, shapes align
    D. empty predicted mask              -> sliding window updates stably

We deliberately do NOT import GLUSForCausalLM / SAM2 / LLaVA here - those
require the full inference stack.  Instead we exercise the two surfaces
this patch actually changed:

    * the strict-shape modulation block from ``model/GLUS.py::evaluate``
      (replicated 1:1 in ``_modulate_text_embeds`` below; if you change the
      guard in GLUS.py please mirror it here)
    * the sliding-window helpers ``_extract_bin_mask_2d`` and
      ``_update_traj_history`` imported directly from ``inference_iter.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from utils.trajectory_utils import (  # noqa: E402
    DEFAULT_HISTORY_LEN,
    STATE_DIM,
)
from model.trajectory_module import TrajectoryEncoder  # noqa: E402


def _modulate_text_embeds(
    text_embeds: torch.Tensor,
    traj_states,
    traj_valid_mask,
    *,
    use_trajectory: bool,
    encoder,
    traj_state_dim: int,
) -> torch.Tensor:
    """Mirror of the in-evaluate() modulation block, no model required."""
    if not (
        use_trajectory
        and encoder is not None
        and traj_states is not None
        and traj_valid_mask is not None
    ):
        return text_embeds
    ts = traj_states
    vm = traj_valid_mask
    if not (
        ts.dim() == 3
        and vm.dim() == 2
        and ts.shape[0] == vm.shape[0]
        and ts.shape[1] == vm.shape[1]
        and ts.shape[-1] == traj_state_dim
        and ts.shape[0] == text_embeds.shape[0]
        and float(vm.sum().item()) > 0.0
    ):
        return text_embeds
    ts_in = ts.to(device=text_embeds.device, dtype=text_embeds.dtype)
    vm_in = vm.to(device=text_embeds.device, dtype=text_embeds.dtype)
    traj_global, _ = encoder(ts_in, vm_in)
    gate = torch.tanh(encoder.alpha).to(
        device=text_embeds.device, dtype=text_embeds.dtype
    )
    delta = traj_global.to(text_embeds.dtype).unsqueeze(1)
    return text_embeds + gate * delta


def _square_mask(H, W, cx, cy, side):
    m = np.zeros((H, W), dtype=np.uint8)
    half = side // 2
    x0 = max(0, cx - half); x1 = min(W, cx + half)
    y0 = max(0, cy - half); y1 = min(H, cy + half)
    m[y0:y1, x0:x1] = 1
    return m


def _load_inference_helpers():
    """Import the two helpers from inference_iter without running its CLI."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "inference_iter_local",
        os.path.join(_REPO_ROOT, "inference_iter.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load inference_iter.py")
    # inference_iter imports heavy deps at module top - we only need the
    # helpers, so we shim out the heavy imports by sys.modules.
    # Instead of re-executing the module, just exec the helper functions:
    src_path = os.path.join(_REPO_ROOT, "inference_iter.py")
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    needle = "def _extract_bin_mask_2d"
    start = src.find(needle)
    if start < 0:
        raise RuntimeError("could not find _extract_bin_mask_2d in inference_iter.py")
    end_marker = "\ndef inference_frames("
    end = src.find(end_marker, start)
    if end < 0:
        raise RuntimeError("could not find end of helpers in inference_iter.py")
    snippet = src[start:end]
    ns = {
        "torch": torch,
        "mask_to_box_state": __import__(
            "utils.trajectory_utils", fromlist=["mask_to_box_state"]
        ).mask_to_box_state,
    }
    exec(compile(snippet, src_path, "exec"), ns)
    return ns["_extract_bin_mask_2d"], ns["_update_traj_history"]


_extract_bin_mask_2d, _update_traj_history = _load_inference_helpers()


# ---------------------------------------------------------------------------
# A. use_trajectory=False  ->  text_embeds is bit-identical
# ---------------------------------------------------------------------------


def test_A_use_trajectory_false_is_identity():
    torch.manual_seed(0)
    enc = TrajectoryEncoder(state_dim=STATE_DIM, hidden_dim=16, out_dim=32)
    text_embeds = torch.randn(1, 1, 32, dtype=torch.float32)
    out = _modulate_text_embeds(
        text_embeds,
        traj_states=torch.zeros(1, DEFAULT_HISTORY_LEN, STATE_DIM),
        traj_valid_mask=torch.ones(1, DEFAULT_HISTORY_LEN),
        use_trajectory=False,
        encoder=enc,
        traj_state_dim=STATE_DIM,
    )
    assert torch.equal(out, text_embeds), "use_trajectory=False must be identity"
    print("[PASS] A. use_trajectory=False -> identity")


# ---------------------------------------------------------------------------
# B. use_trajectory=True, empty history  ->  inference_iter passes None
#    and evaluate() short-circuits.  Here we simulate exactly that.
# ---------------------------------------------------------------------------


def test_B_empty_history_passes_none_and_skips():
    torch.manual_seed(0)
    enc = TrajectoryEncoder(state_dim=STATE_DIM, hidden_dim=16, out_dim=32)
    text_embeds = torch.randn(1, 1, 32, dtype=torch.float32)
    # inference_iter.py builds: "len(traj_history) == 0 -> traj_*_in = None".
    out = _modulate_text_embeds(
        text_embeds,
        traj_states=None,
        traj_valid_mask=None,
        use_trajectory=True,
        encoder=enc,
        traj_state_dim=STATE_DIM,
    )
    assert torch.equal(out, text_embeds), "empty history must skip modulation"
    print("[PASS] B. empty history -> evaluate() skips, text_embeds unchanged")


# ---------------------------------------------------------------------------
# C. use_trajectory=True, real history  ->  modulation fires, shape preserved
# ---------------------------------------------------------------------------


def test_C_real_history_modulates_and_preserves_shape():
    torch.manual_seed(0)
    enc = TrajectoryEncoder(state_dim=STATE_DIM, hidden_dim=16, out_dim=32)
    # Force a non-zero gate so the modulation is observable; the production
    # path starts from a trained checkpoint which may already have alpha!=0.
    with torch.no_grad():
        enc.alpha.fill_(0.5)
    text_embeds = torch.randn(1, 1, 32, dtype=torch.float32)
    traj_states = torch.randn(1, DEFAULT_HISTORY_LEN, STATE_DIM)
    traj_valid_mask = torch.zeros(1, DEFAULT_HISTORY_LEN)
    traj_valid_mask[0, -3:] = 1.0  # last 3 slots valid - simulates 3 prev preds
    out = _modulate_text_embeds(
        text_embeds,
        traj_states=traj_states,
        traj_valid_mask=traj_valid_mask,
        use_trajectory=True,
        encoder=enc,
        traj_state_dim=STATE_DIM,
    )
    assert out.shape == text_embeds.shape, (
        f"shape changed: {tuple(out.shape)} vs {tuple(text_embeds.shape)}"
    )
    diff = (out - text_embeds).abs().max().item()
    assert diff > 0.0, "modulation should change text_embeds at alpha=0.5"
    assert torch.isfinite(out).all()
    print(f"[PASS] C. real history -> modulation fires (Δmax={diff:.4g})")


def test_C_strict_shape_mismatch_skips():
    """row count of traj_states != text_embeds.shape[0] must skip (no broadcast)."""
    torch.manual_seed(0)
    enc = TrajectoryEncoder(state_dim=STATE_DIM, hidden_dim=16, out_dim=32)
    with torch.no_grad():
        enc.alpha.fill_(0.5)
    text_embeds = torch.randn(2, 1, 32, dtype=torch.float32)  # 2 rows
    traj_states = torch.randn(1, DEFAULT_HISTORY_LEN, STATE_DIM)  # 1 row
    traj_valid_mask = torch.ones(1, DEFAULT_HISTORY_LEN)
    out = _modulate_text_embeds(
        text_embeds,
        traj_states=traj_states,
        traj_valid_mask=traj_valid_mask,
        use_trajectory=True,
        encoder=enc,
        traj_state_dim=STATE_DIM,
    )
    assert torch.equal(out, text_embeds), (
        "row-count mismatch must skip modulation (no broadcast)"
    )
    print("[PASS] C'. shape mismatch (B=1 vs N=2) -> skip + warn (no broadcast)")


def test_C_all_pad_history_skips():
    torch.manual_seed(0)
    enc = TrajectoryEncoder(state_dim=STATE_DIM, hidden_dim=16, out_dim=32)
    with torch.no_grad():
        enc.alpha.fill_(0.5)
    text_embeds = torch.randn(1, 1, 32, dtype=torch.float32)
    out = _modulate_text_embeds(
        text_embeds,
        traj_states=torch.zeros(1, DEFAULT_HISTORY_LEN, STATE_DIM),
        traj_valid_mask=torch.zeros(1, DEFAULT_HISTORY_LEN),  # ALL pad
        use_trajectory=True,
        encoder=enc,
        traj_state_dim=STATE_DIM,
    )
    assert torch.equal(out, text_embeds), "all-pad valid mask must skip"
    print("[PASS] C''. all-pad valid mask -> skip")


# ---------------------------------------------------------------------------
# D. Empty predicted mask  ->  sliding window update is stable
# ---------------------------------------------------------------------------


def test_D_empty_mask_updates_window_stably():
    H, W = 64, 64
    traj_history = []
    prev_valid_state = None
    T_h = DEFAULT_HISTORY_LEN

    # Step 1: non-empty mask
    m1 = _square_mask(H, W, 24, 32, 10)
    bin1 = _extract_bin_mask_2d(torch.from_numpy(m1).unsqueeze(0))  # [1,H,W]
    assert bin1.dim() == 2 and bin1.dtype == torch.bool
    prev_valid_state = _update_traj_history(
        bin1, traj_history, prev_valid_state, T_h, STATE_DIM
    )
    assert len(traj_history) == 1
    assert float(traj_history[-1][9]) == 1.0, "q must be 1 for non-empty mask"
    assert prev_valid_state is not None

    saved_prev = prev_valid_state.clone()

    # Step 2: EMPTY mask  (this is the failure / no-detection case)
    m2 = np.zeros((H, W), dtype=np.uint8)
    bin2 = _extract_bin_mask_2d(torch.from_numpy(m2))               # [H,W]
    assert bin2.dim() == 2
    prev_valid_state = _update_traj_history(
        bin2, traj_history, prev_valid_state, T_h, STATE_DIM
    )
    assert len(traj_history) == 2
    assert float(traj_history[-1][9]) == 0.0, "q must be 0 for empty mask"
    assert torch.equal(prev_valid_state, saved_prev), (
        "prev_valid_state must NOT update on empty mask"
    )

    # Step 3: non-empty again - velocities are computed vs step 1 (last q>0).
    m3 = _square_mask(H, W, 36, 32, 10)
    bin3 = _extract_bin_mask_2d(torch.from_numpy(m3))
    prev_valid_state = _update_traj_history(
        bin3, traj_history, prev_valid_state, T_h, STATE_DIM
    )
    assert len(traj_history) == 3
    vx_step3 = float(traj_history[-1][6])
    assert vx_step3 != 0.0, "non-empty mask after empty must produce non-zero vx"
    print(
        f"[PASS] D. empty-mask window update stable "
        f"(history len={len(traj_history)}, vx@step3={vx_step3:.4f})"
    )


def test_D_overflow_drops_oldest():
    H, W = 64, 64
    traj_history = []
    prev_valid_state = None
    T_h = 3
    for k in range(5):
        m = _square_mask(H, W, 20 + 2 * k, 32, 8)
        bin_m = _extract_bin_mask_2d(torch.from_numpy(m).unsqueeze(0))
        prev_valid_state = _update_traj_history(
            bin_m, traj_history, prev_valid_state, T_h, STATE_DIM
        )
    assert len(traj_history) == T_h, (
        f"expected window len {T_h}, got {len(traj_history)}"
    )
    print(f"[PASS] D'. overflow drops oldest, window kept at T_h={T_h}")


def test_D_extract_handles_extra_leading_dims():
    """Defensive shape collapse: [1,1,H,W] -> [H,W]."""
    H, W = 16, 16
    m = np.eye(H, dtype=np.uint8)
    bin_m = _extract_bin_mask_2d(torch.from_numpy(m)[None, None, :, :])
    assert bin_m.shape == (H, W) and bin_m.dtype == torch.bool
    print("[PASS] D''. _extract_bin_mask_2d collapses [1,1,H,W] -> [H,W]")


if __name__ == "__main__":
    tests = [
        test_A_use_trajectory_false_is_identity,
        test_B_empty_history_passes_none_and_skips,
        test_C_real_history_modulates_and_preserves_shape,
        test_C_strict_shape_mismatch_skips,
        test_C_all_pad_history_skips,
        test_D_empty_mask_updates_window_stably,
        test_D_overflow_drops_oldest,
        test_D_extract_handles_extra_leading_dims,
    ]
    failed = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"[FAIL] {fn.__name__}: {exc}")
            traceback.print_exc()
            failed.append(fn.__name__)
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed.")
    if failed:
        raise SystemExit(1)
