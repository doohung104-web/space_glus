"""Phase-1 trajectory encoder for GLUS.

Pipeline (per design ``trajectory_design_for_glus.md`` §4.2):

    [N, T, state_dim]
        -> input_mlp        -> [N, T, hidden_dim]
        -> 2-layer GRU      -> [N, T, hidden_dim]
        -> masked mean pool + last-valid hidden -> concat -> [N, 2*hidden_dim]
        -> proj             -> traj_global [N, out_dim]
        -> forecast_head    -> traj_pred   [N, state_dim]

The module owns a single learnable scalar gate ``alpha`` initialised to 0
(so that ``tanh(alpha) == 0``).  This guarantees that, at the very start of
training, the trajectory branch contributes nothing to the SEG embedding and
the original GLUS forward is recovered exactly.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    """Encode ``T``-step trajectory states into a global condition vector.

    Args:
        state_dim:    dimension of the per-frame state vector (default 10).
        hidden_dim:   internal hidden dim of MLP / GRU.
        out_dim:      target dim of ``traj_global`` (typically the GLUS
                      ``out_dim``, i.e. the SEG embedding dim).
        encoder_type: ``'gru'`` (default).  ``'mamba'`` is reserved for a
                      future phase and currently raises ``NotImplementedError``.
        num_layers:   number of GRU layers (default 2).
        dropout:      dropout *between* GRU layers (PyTorch convention; only
                      effective when ``num_layers > 1``).
    """

    def __init__(
        self,
        state_dim: int = 10,
        hidden_dim: int = 256,
        out_dim: int = 256,
        encoder_type: str = "gru",
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.encoder_type = encoder_type.lower()

        self.input_mlp = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        if self.encoder_type == "gru":
            self.temporal = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=int(num_layers),
                batch_first=True,
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            )
        elif self.encoder_type == "mamba":
            raise NotImplementedError(
                "encoder_type='mamba' is reserved for a future phase; "
                "use 'gru' for the Phase-1 MVP."
            )
        else:
            raise ValueError(f"unknown encoder_type: {encoder_type!r}")

        # masked mean + last-valid step -> 2 * hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )

        # Init-0 gate so that the SEG modulation is a no-op at step 0.
        self.alpha = nn.Parameter(torch.zeros(()))

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``x: [N, T, D]``, ``valid: [N, T]`` -> ``[N, D]`` (zero when all pad)."""
        v = valid.unsqueeze(-1).to(x.dtype)
        s = (x * v).sum(dim=1)
        denom = v.sum(dim=1).clamp(min=1.0)
        return s / denom

    @staticmethod
    def _last_valid(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Gather the last valid time-step per row.  All-pad -> zeros."""
        N, T, D = x.shape
        any_valid = (valid.sum(dim=1) > 0)
        flipped = torch.flip(valid > 0, dims=[1]).to(torch.int32)
        last_from_right = flipped.argmax(dim=1)
        last_idx = (T - 1 - last_from_right).clamp(min=0)
        rows = torch.arange(N, device=x.device)
        gathered = x[rows, last_idx]
        gathered = gathered * any_valid.unsqueeze(-1).to(x.dtype)
        return gathered

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(
        self,
        traj_states: torch.Tensor,
        traj_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode trajectory states.

        Args:
            traj_states:     ``[N, T, state_dim]`` float tensor.
            traj_valid_mask: ``[N, T]`` 0/1 mask.
        Returns:
            traj_global: ``[N, out_dim]`` condition vector for SEG modulation.
            traj_pred:   ``[N, state_dim]`` next-state prediction (forecast).
        """
        if traj_states.dim() != 3 or traj_states.shape[-1] != self.state_dim:
            raise ValueError(
                f"traj_states must be [N, T, {self.state_dim}], "
                f"got {tuple(traj_states.shape)}"
            )
        if traj_valid_mask.shape != traj_states.shape[:2]:
            raise ValueError(
                "traj_valid_mask shape mismatch: "
                f"states={tuple(traj_states.shape)}, "
                f"mask={tuple(traj_valid_mask.shape)}"
            )

        x = self.input_mlp(traj_states)
        # GRU is non-deterministic w.r.t. mask, but downstream pooling uses
        # the mask explicitly so padding only contaminates the hidden state
        # of *its own* time step.
        x, _ = self.temporal(x)

        mean_feat = self._masked_mean(x, traj_valid_mask)
        last_feat = self._last_valid(x, traj_valid_mask)
        pooled = torch.cat([mean_feat, last_feat], dim=-1)

        traj_global = self.proj(pooled)
        traj_pred = self.forecast_head(pooled)
        return traj_global, traj_pred

    # Diagnostic utility used by tests / training logs.
    def gate(self) -> torch.Tensor:
        return torch.tanh(self.alpha)
