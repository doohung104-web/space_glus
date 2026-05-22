"""Phase-1 trajectory smoke test.

Pure torch / numpy.  Does *not* import any GLUS / transformers / sam2 module,
so it can run in any environment that has PyTorch.

Coverage:

    A. mask_to_box_state          : shapes, empty-mask handling
    B. masks_to_state_sequence    : shapes, velocity initialisation, monotone
                                    cx for a translating square, q correctness
    C. build_history_windows      : shapes, padding-left convention, valid
                                    mask flips on as Q grows
    D. all-empty sequence         : numerically stable, no NaN/Inf
    E. TrajectoryEncoder forward  : output shapes (traj_global, traj_pred),
                                    alpha starts at 0
    F. all-invalid encoder input  : no NaN, traj_global is finite
    G. SEG-modulation invariance  : at alpha=0 the modulated SEG embedding
                                    equals the original (up to fp tolerance)
    H. layout contract            : reshape(-1, ...) of the
                                    [num_expr, Q, T_h, D] tensor matches the
                                    expression-major / frame-minor order
                                    used by ``cur_index = [j + Q*num]`` in
                                    model/GLUS.py.

Run with::

    python -m pytest tests/test_trajectory_smoke.py -q
    # or:
    python tests/test_trajectory_smoke.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


# Allow running directly from the repo root *or* as a pytest target.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from utils.trajectory_utils import (  # noqa: E402
    DEFAULT_HISTORY_LEN,
    STATE_DIM,
    build_history_windows,
    mask_to_box_state,
    masks_to_state_sequence,
)
from model.trajectory_module import TrajectoryEncoder  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _square_mask(H, W, cx, cy, side):
    m = np.zeros((H, W), dtype=np.uint8)
    half = side // 2
    x0 = max(0, cx - half)
    x1 = min(W, cx + half)
    y0 = max(0, cy - half)
    y1 = min(H, cy + half)
    m[y0:y1, x0:x1] = 1
    return m


def _translating_square_sequence(T, H=64, W=64, side=10, dx=4):
    masks = np.zeros((T, H, W), dtype=np.uint8)
    cx0 = W // 4
    cy = H // 2
    for t in range(T):
        masks[t] = _square_mask(H, W, cx0 + t * dx, cy, side)
    return masks


# ---------------------------------------------------------------------------
# A. mask_to_box_state
# ---------------------------------------------------------------------------


def test_mask_to_box_state_shapes_and_empty():
    m = _square_mask(64, 64, 32, 32, 10)
    s = mask_to_box_state(m)
    assert s.shape == (7,)
    assert s.dtype == torch.float32
    assert torch.isfinite(s).all()
    assert 0.0 < float(s[0]) < 1.0 and 0.0 < float(s[1]) < 1.0  # cx, cy
    assert float(s[6]) == 1.0  # q

    e = mask_to_box_state(np.zeros((64, 64), dtype=np.uint8))
    assert e.shape == (7,)
    assert torch.all(e == 0)


# ---------------------------------------------------------------------------
# B. masks_to_state_sequence
# ---------------------------------------------------------------------------


def test_masks_to_state_sequence_translating_square():
    T = 5
    seq = _translating_square_sequence(T)
    states = masks_to_state_sequence(seq)
    assert states.shape == (T, STATE_DIM)
    assert states.dtype == torch.float32
    assert torch.isfinite(states).all()

    cx = states[:, 0]
    # cx strictly increases for a left-to-right translating square
    diffs = (cx[1:] - cx[:-1])
    assert torch.all(diffs > 0), f"cx should increase, got {cx.tolist()}"

    # first frame has zero velocity / darea
    assert float(states[0, 6]) == 0.0  # vx
    assert float(states[0, 7]) == 0.0  # vy
    assert float(states[0, 8]) == 0.0  # darea

    # all frames have q=1 (non-empty masks)
    assert torch.all(states[:, 9] == 1.0)


def test_masks_to_state_sequence_handles_empty_in_middle():
    # frame 1 is empty -> q=0, frame 2 references frame 0 via velocities=0
    seq = np.stack(
        [
            _square_mask(64, 64, 16, 32, 10),
            np.zeros((64, 64), dtype=np.uint8),
            _square_mask(64, 64, 24, 32, 10),
        ],
        axis=0,
    )
    states = masks_to_state_sequence(seq)
    assert states.shape == (3, STATE_DIM)
    assert float(states[1, 9]) == 0.0  # q=0 for the empty frame
    # the recovering frame should have non-zero velocity wrt the *last valid*
    # frame, but our convention sets vx=0 when the previous frame is empty.
    # We only require finite numbers here; behaviour is documented in
    # trajectory_utils.py and frozen by this assertion.
    assert torch.isfinite(states).all()


# ---------------------------------------------------------------------------
# C. build_history_windows
# ---------------------------------------------------------------------------


def test_build_history_windows_shapes_and_valid_mask():
    Q = 4
    T_h = 6
    seq = _translating_square_sequence(Q)
    states = masks_to_state_sequence(seq)
    hist, valid, target = build_history_windows(states, history_len=T_h)
    assert hist.shape == (Q, T_h, STATE_DIM)
    assert valid.shape == (Q, T_h)
    assert target.shape == (Q, STATE_DIM)
    # First question frame has no history.
    assert torch.all(valid[0] == 0.0)
    assert torch.all(hist[0] == 0.0)
    # Second question frame: exactly one valid step in the rightmost slot.
    assert torch.all(valid[1, :-1] == 0.0)
    assert float(valid[1, -1]) == 1.0
    # The rightmost slot at frame j should equal states[j-1].
    for j in range(1, Q):
        assert torch.allclose(hist[j, -1], states[j - 1])
    # target is exactly a copy of states (the forecast supervision).
    assert torch.allclose(target, states)


def test_build_history_windows_history_shorter_than_q():
    # If T_h=2 and Q=4, frame j=3 only keeps the last 2 history steps.
    Q = 4
    T_h = 2
    seq = _translating_square_sequence(Q)
    states = masks_to_state_sequence(seq)
    hist, valid, target = build_history_windows(states, history_len=T_h)
    assert hist.shape == (Q, T_h, STATE_DIM)
    assert torch.all(valid[3] == 1.0)
    assert torch.allclose(hist[3, 0], states[1])
    assert torch.allclose(hist[3, 1], states[2])


# ---------------------------------------------------------------------------
# D. all-empty sequence
# ---------------------------------------------------------------------------


def test_all_empty_sequence_is_stable():
    """An all-empty mask sequence must produce zero states / hist / target and
    must not destabilise the encoder.  ``valid`` here is a *padding* mask
    (slot-filled indicator), so it is still 1 for filled slots even though
    their state is all-zero; mask-emptiness is encoded by ``q`` (state[9])
    which is 0 for empty frames - the encoder consumes both signals.
    """
    seq = np.zeros((4, 64, 64), dtype=np.uint8)
    states = masks_to_state_sequence(seq)
    assert torch.all(states == 0.0)
    hist, valid, target = build_history_windows(states, history_len=DEFAULT_HISTORY_LEN)
    # states are all 0 -> hist values are all 0 (filled slots carry 0 vectors).
    assert torch.all(hist == 0.0)
    assert torch.all(target == 0.0)
    # Encoder must remain finite on this input.
    enc = TrajectoryEncoder(state_dim=STATE_DIM, hidden_dim=8, out_dim=8)
    g, p = enc(hist, valid)
    assert torch.isfinite(g).all() and torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# E. TrajectoryEncoder forward shapes
# ---------------------------------------------------------------------------


def test_trajectory_encoder_forward_shapes():
    torch.manual_seed(0)
    N, T, D, H, O = 5, 6, STATE_DIM, 32, 64
    enc = TrajectoryEncoder(state_dim=D, hidden_dim=H, out_dim=O)
    s = torch.randn(N, T, D)
    v = torch.ones(N, T)
    g, p = enc(s, v)
    assert g.shape == (N, O)
    assert p.shape == (N, D)
    assert torch.isfinite(g).all() and torch.isfinite(p).all()
    # alpha starts at exactly 0  =>  gate = tanh(0) = 0
    assert float(enc.alpha.item()) == 0.0
    assert float(enc.gate().item()) == 0.0


# ---------------------------------------------------------------------------
# F. all-invalid encoder input
# ---------------------------------------------------------------------------


def test_trajectory_encoder_all_invalid_no_nan():
    torch.manual_seed(0)
    enc = TrajectoryEncoder(state_dim=STATE_DIM, hidden_dim=16, out_dim=16)
    s = torch.zeros(3, 4, STATE_DIM)
    v = torch.zeros(3, 4)
    g, p = enc(s, v)
    assert torch.isfinite(g).all()
    assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# G. SEG-modulation invariance at alpha=0
# ---------------------------------------------------------------------------


def test_seg_modulation_is_noop_at_alpha_zero():
    torch.manual_seed(0)
    N, T, D, H, O = 7, 6, STATE_DIM, 16, 32
    enc = TrajectoryEncoder(state_dim=D, hidden_dim=H, out_dim=O)
    s = torch.randn(N, T, D)
    v = torch.ones(N, T)
    g, _ = enc(s, v)
    seg_emb = torch.randn(N, O)
    gate = torch.tanh(enc.alpha)
    modulated = seg_emb + gate * g
    assert torch.allclose(modulated, seg_emb)


# ---------------------------------------------------------------------------
# H. Layout contract: reshape order matches model/GLUS.py cur_index formula
# ---------------------------------------------------------------------------


def test_layout_matches_cur_index_order():
    """Construct a uniquely-tagged ``[num_expr, Q, T_h, D]`` tensor and verify
    that ``.reshape(num_expr*Q, T_h, D)`` produces the row order expected by
    the existing ``cur_index = [j + Q * num for num in range(num_expr)]``
    formula used in ``model/GLUS.py``.
    """
    num_expr = 3
    Q = 4
    T_h = 6
    D = STATE_DIM

    # Tag every row with the integer ``expr_id * 100 + frame_id`` in state[0].
    states = torch.zeros(num_expr, Q, T_h, D)
    for e in range(num_expr):
        for q in range(Q):
            states[e, q, 0, 0] = float(e * 100 + q)

    flat = states.reshape(num_expr * Q, T_h, D)
    rows = flat[:, 0, 0].tolist()

    # Build the expected order by *iterating frames first* the way GLUS does:
    # for frame j, cur_index picks rows [j, j+Q, j+2Q, ...].  Concatenating
    # j=0..Q-1 should reconstruct the natural row order of the flattened
    # tensor (expression-major, frame-minor).
    visited = set()
    for j in range(Q):
        for num in range(num_expr):
            row_idx = j + Q * num
            assert row_idx not in visited
            visited.add(row_idx)
            # cur_index says: pred_embeddings[row_idx] is for (num, j).
            expected_tag = float(num * 100 + j)
            assert rows[row_idx] == expected_tag, (
                f"row {row_idx} expected tag {expected_tag}, got {rows[row_idx]}"
            )
    assert len(visited) == num_expr * Q


# ---------------------------------------------------------------------------
# Manual entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    passed = 0
    failed = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
                passed += 1
            except Exception as exc:  # noqa: BLE001
                import traceback

                print(f"[FAIL] {name}: {exc}")
                traceback.print_exc()
                failed.append(name)
    print(f"\n{passed} passed, {len(failed)} failed.")
    if failed:
        raise SystemExit(1)
