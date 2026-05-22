"""Phase-1 trajectory-state utilities (see ``trajectory_design_for_glus.md``).

This module only exposes pure functions that turn binary masks into a
low-dimensional motion-state sequence and build history windows over a
``question_frame_num`` window.  No model dependency, no I/O - everything is
torch / numpy only so it can be unit-tested in isolation.

State convention (10 dims):

    s_t = [cx, cy, w, h, area, ratio, vx, vy, darea, q]

    * ``cx, cy``       - bbox center,   normalised by image width / height
    * ``w,  h``        - bbox size,     normalised by image width / height
    * ``area``         - mask area ratio (>=0)
    * ``ratio``        - clamp(log(w_px / h_px), -2, 2)
    * ``vx, vy``       - cx/cy delta to previous (valid) frame, clipped to [-1,1]
    * ``darea``        - area delta to previous (valid) frame, clipped to [-1,1]
    * ``q``            - 1 if the mask is non-empty, else 0

Important Phase-1 constraint: history windows are constructed *only* from
the mask sequence the dataset already exposes (i.e. the ``question_frame_num``
GT masks).  We do **not** sample extra frames outside that window.
"""

from __future__ import annotations

from typing import Tuple, Union

import math

import numpy as np
import torch


STATE_DIM: int = 10
DEFAULT_HISTORY_LEN: int = 6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_tensor(mask: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Coerce input to a 2-D torch boolean tensor on CPU (no copy when possible)."""
    if isinstance(mask, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(mask))
    elif isinstance(mask, torch.Tensor):
        t = mask
    else:
        raise TypeError(f"mask must be ndarray or Tensor, got {type(mask)}")
    if t.dim() != 2:
        raise ValueError(f"mask must be 2-D, got shape {tuple(t.shape)}")
    return t > 0


def _bbox_from_binary_mask(mask_bool: torch.Tensor) -> Tuple[float, float, float, float, bool]:
    """Return ``(x_min, y_min, w_px, h_px, non_empty)`` in pixel units."""
    if mask_bool.sum() <= 0:
        return 0.0, 0.0, 0.0, 0.0, False
    ys, xs = torch.where(mask_bool)
    x_min = float(xs.min().item())
    x_max = float(xs.max().item())
    y_min = float(ys.min().item())
    y_max = float(ys.max().item())
    w_px = max(1.0, x_max - x_min + 1.0)
    h_px = max(1.0, y_max - y_min + 1.0)
    return x_min, y_min, w_px, h_px, True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mask_to_box_state(
    mask: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the 6 static state components plus ``q`` for a single mask.

    Args:
        mask: ``[H, W]`` binary mask (numpy / torch, any integer / bool dtype).
    Returns:
        ``state[7] = [cx, cy, w, h, area, ratio, q]`` as a float32 tensor.
        ``cx, cy, w, h`` are normalised to ``[0, 1]``.  Empty mask -> all 0
        and ``q = 0``.  ``ratio`` is ``clamp(log(w_px / h_px), -2, 2)`` and 0
        for empty masks.
    """
    m = _ensure_tensor(mask)
    H, W = m.shape
    img_area = float(max(1, H * W))
    fH = float(max(1, H))
    fW = float(max(1, W))

    x_min, y_min, w_px, h_px, valid = _bbox_from_binary_mask(m)
    if not valid:
        return torch.zeros(7, dtype=torch.float32)

    cx = (x_min + w_px / 2.0) / fW
    cy = (y_min + h_px / 2.0) / fH
    w = w_px / fW
    h = h_px / fH
    area = float(m.sum().item()) / img_area
    ratio = math.log(max(w_px, eps) / max(h_px, eps))
    ratio = max(-2.0, min(2.0, ratio))

    return torch.tensor([cx, cy, w, h, area, ratio, 1.0], dtype=torch.float32)


def masks_to_state_sequence(
    masks: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """Turn a ``[T, H, W]`` mask sequence into a ``[T, STATE_DIM]`` state matrix.

    Velocities (``vx, vy, darea``) are computed as plain forward differences
    w.r.t. the *previous valid* frame.  When the previous frame was empty
    (``q=0``) we set the velocity components of the current frame to 0 to
    avoid spurious jumps, but the current frame's ``q`` stays 1 (its bbox is
    still measurable).  The very first frame always has zero velocities.

    Args:
        masks: ``[T, H, W]`` binary mask sequence.
    Returns:
        states: ``[T, STATE_DIM]`` float32 tensor.
    """
    if isinstance(masks, np.ndarray):
        masks_t = torch.from_numpy(np.ascontiguousarray(masks))
    elif isinstance(masks, torch.Tensor):
        masks_t = masks
    else:
        raise TypeError(f"masks must be ndarray or Tensor, got {type(masks)}")
    if masks_t.dim() != 3:
        raise ValueError(f"masks must be 3-D, got shape {tuple(masks_t.shape)}")

    T = int(masks_t.shape[0])
    states = torch.zeros(T, STATE_DIM, dtype=torch.float32)
    prev_idx: int = -1

    for t in range(T):
        s7 = mask_to_box_state(masks_t[t])
        cx, cy, w, h, area, ratio, q = (float(v) for v in s7.tolist())

        if q > 0 and prev_idx >= 0:
            p = states[prev_idx]
            vx = max(-1.0, min(1.0, cx - float(p[0].item())))
            vy = max(-1.0, min(1.0, cy - float(p[1].item())))
            darea = max(-1.0, min(1.0, area - float(p[4].item())))
        else:
            vx = 0.0
            vy = 0.0
            darea = 0.0

        states[t] = torch.tensor(
            [cx, cy, w, h, area, ratio, vx, vy, darea, q], dtype=torch.float32
        )
        if q > 0:
            prev_idx = t

    return states


def build_history_windows(
    states: torch.Tensor,
    history_len: int = DEFAULT_HISTORY_LEN,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-frame history windows from a Q-frame state sequence.

    For the ``j``-th question frame the history is composed of the previous
    ``history_len`` *question-window* states (clipped at the start of the
    sequence, with left-side zero padding).  The current frame's state is
    returned separately as the supervision target for the forecast loss.

    Args:
        states:      ``[Q, STATE_DIM]`` float tensor.
        history_len: ``T_h`` (default 6).
    Returns:
        hist:   ``[Q, T_h, STATE_DIM]`` (left-padded with zeros).
        valid:  ``[Q, T_h]`` 1 = real, 0 = padding.
        target: ``[Q, STATE_DIM]`` (just a copy of ``states``).
    """
    if states.dim() != 2 or states.shape[-1] != STATE_DIM:
        raise ValueError(
            f"states must be [Q, {STATE_DIM}], got {tuple(states.shape)}"
        )
    if history_len <= 0:
        raise ValueError(f"history_len must be > 0, got {history_len}")

    Q = int(states.shape[0])
    hist = torch.zeros(Q, history_len, STATE_DIM, dtype=states.dtype)
    valid = torch.zeros(Q, history_len, dtype=torch.float32)
    target = states.clone()

    for j in range(Q):
        # Most recent history sits at the *rightmost* slot, matching the
        # natural reading order of the temporal encoder.
        n_real = min(history_len, j)
        if n_real > 0:
            hist[j, -n_real:] = states[j - n_real:j]
            valid[j, -n_real:] = 1.0
    return hist, valid, target
