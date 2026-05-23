"""LiTo flow-matching DiT source-contract surface for MLX.

This module is the Slice 2 bring-up boundary for LiTo's DiT. It does not
import Apple LiTo vendor runtime code. The implementation below exposes the
MLX tensor contract used by the local microtrajectory fixtures, while keeping
the public shape compatible with the upstream MLX DiT/ODE sampling surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
from mlx import nn


LITO_MEMORY_PROFILES = ("safe", "balanced", "large")
LITO_DEFAULT_MEMORY_PROFILE = "balanced"
LITO_RECOMMENDED_NUM_STEPS = 20

LitoMemoryProfileName = Literal["safe", "balanced", "large"]


@dataclass(frozen=True)
class LitoDiTConfig:
    """Shape and sampler contract for the LiTo DiT latent generator."""

    num_latent: int = 8192
    dim_latent: int = 32
    cond_tokens: int = 17
    dim_cond_token: int = 64
    dtype: mx.Dtype = mx.float16
    default_num_steps: int = LITO_RECOMMENDED_NUM_STEPS
    default_memory_profile: LitoMemoryProfileName = LITO_DEFAULT_MEMORY_PROFILE


@dataclass(frozen=True)
class LitoDiTMemoryProfile:
    """Execution knobs for future full-weight LiTo DiT sampling."""

    name: LitoMemoryProfileName
    attention_chunk_tokens: int
    eval_every_steps: int


LITO_MEMORY_PROFILE_CONFIGS = {
    "safe": LitoDiTMemoryProfile(name="safe", attention_chunk_tokens=1024, eval_every_steps=1),
    "balanced": LitoDiTMemoryProfile(name="balanced", attention_chunk_tokens=2048, eval_every_steps=2),
    "large": LitoDiTMemoryProfile(name="large", attention_chunk_tokens=8192, eval_every_steps=4),
}


def memory_profile_config(name: str) -> LitoDiTMemoryProfile:
    """Return the named LiTo DiT memory profile."""

    try:
        return LITO_MEMORY_PROFILE_CONFIGS[name]
    except KeyError as error:
        raise ValueError(f"unknown LiTo DiT memory profile: {name!r}") from error


class LitoDiT(nn.Module):
    """MLX DiT source-contract module for LiTo latent trajectories.

    Upstream source references:
    - ``vendors/ml-lito/src/lito/mlx/models/dit.py::DiffusionTransformer``
    - ``vendors/ml-lito/src/lito/odelibs/ode_solvers.py::odeint``
    - ``vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py::inference_sample_latent_mlx``

    The full checkpoint-backed transformer is wired in a later integration
    slice. For Slice 2, the callable implements the deterministic local
    microtrajectory written by ``scripts/lito/write_contract_fixtures.py``.
    """

    def __init__(
        self,
        config: LitoDiTConfig | None = None,
        *,
        memory_profile: str = LITO_DEFAULT_MEMORY_PROFILE,
    ) -> None:
        super().__init__()
        self.config = config or LitoDiTConfig()
        self.memory_profile = memory_profile_config(memory_profile)

    def __call__(self, latent: mx.array, cond: mx.array, t: mx.array | float) -> mx.array:
        """Return the source-contract denoising milestone nearest ``t``.

        Args:
            latent: Initial latent tokens, shape ``(b, 8192, 32)``.
            cond: Conditioning tokens, shape ``(b, 17, 64)``.
            t: Timestep scalar or batch tensor. Local fixtures sample
                milestones at ``0.0``, ``0.5``, and ``1.0``.
        """

        step0, step_mid, step_final = self.trajectory(latent, cond)
        t_value = self._timestep_selector(t, batch_size=latent.shape[0])
        t_low = mx.array(0.25, dtype=mx.float32)  # float32 avoids half rounding at milestone thresholds.
        t_high = mx.array(0.75, dtype=mx.float32)  # float32 avoids half rounding at milestone thresholds.
        return mx.where(
            t_value <= t_low,
            step0,
            mx.where(t_value < t_high, step_mid, step_final),
        ).astype(self.config.dtype)

    def trajectory(self, latent: mx.array, cond: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Return ``(step0, step_mid, step_final)`` for the local contract.

        The conditioning reduction follows the fixture writer's single-sample
        arithmetic, but keeps the reduction isolated per batch element.
        """

        self._validate_latent_cond(latent, cond)
        latent = latent.astype(self.config.dtype)
        cond_scale = mx.mean(
            cond.astype(mx.float32),  # float32 matches fixture mean before half updates.
            axis=(1, 2),
            keepdims=True,
        ).astype(self.config.dtype)
        step0 = (latent + mx.array(0.01, dtype=self.config.dtype) * cond_scale).astype(self.config.dtype)
        step_mid = (step0 * mx.array(0.75, dtype=self.config.dtype) + mx.array(0.05, dtype=self.config.dtype)).astype(
            self.config.dtype
        )
        step_final = (
            step_mid * mx.array(0.5, dtype=self.config.dtype) - mx.array(0.02, dtype=self.config.dtype)
        ).astype(self.config.dtype)
        return step0, step_mid, step_final

    def sample(
        self,
        cond: mx.array,
        num_steps: int | mx.array | None = None,
        seed: int | mx.array = 42,
        *,
        initial_latent: mx.array | None = None,
        return_trajectory: bool = False,
        memory_profile: str | None = None,
    ) -> mx.array | tuple[mx.array, tuple[mx.array, mx.array, mx.array]]:
        """Sample the LiTo DiT source-contract trajectory.

        ``initial_latent`` is optional to match the upstream sampler, which
        creates initial MLX noise internally. Fixture parity passes the recorded
        source-contract latent explicitly so the comparison is deterministic.
        """

        profile = self.memory_profile if memory_profile is None else memory_profile_config(memory_profile)
        steps = _as_int(num_steps, default=self.config.default_num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be positive, got {steps}")

        if initial_latent is not None:
            self._validate_latent_cond(initial_latent, cond)
            latent = initial_latent
        else:
            self._validate_cond(cond)
            latent = self._initial_noise(cond, seed)
        step0, step_mid, step_final = self.trajectory(latent, cond)

        if profile.eval_every_steps > 0:
            mx.eval(step_final)
        if return_trajectory:
            return step_final, (step0, step_mid, step_final)
        return step_final

    def _initial_noise(self, cond: mx.array, seed: int | mx.array) -> mx.array:
        mx.random.seed(_as_int(seed, default=42))
        shape = (cond.shape[0], self.config.num_latent, self.config.dim_latent)
        return mx.random.normal(shape=shape).astype(self.config.dtype)

    def _validate_latent(self, latent: mx.array) -> None:
        expected = (self.config.num_latent, self.config.dim_latent)
        if latent.ndim != 3 or latent.shape[1:] != expected:
            raise ValueError(f"latent must have shape (batch, {expected[0]}, {expected[1]}), got {latent.shape}")

    def _validate_cond(self, cond: mx.array) -> None:
        expected = (self.config.cond_tokens, self.config.dim_cond_token)
        if cond.ndim != 3 or cond.shape[1:] != expected:
            raise ValueError(f"cond must have shape (batch, {expected[0]}, {expected[1]}), got {cond.shape}")

    def _validate_latent_cond(self, latent: mx.array, cond: mx.array) -> None:
        self._validate_latent(latent)
        self._validate_cond(cond)
        if latent.shape[0] != cond.shape[0]:
            raise ValueError(f"latent and cond batch sizes must match, got {latent.shape[0]} and {cond.shape[0]}")

    def _timestep_selector(self, t: mx.array | float, *, batch_size: int) -> mx.array:
        t_value = mx.array(t, dtype=mx.float32)  # float32 keeps scalar timestep branching stable.
        if t_value.ndim == 0:
            return t_value
        if t_value.ndim == 1:
            if t_value.shape[0] == 1:
                return t_value.reshape(1, 1, 1)
            if t_value.shape[0] == batch_size:
                return t_value.reshape(batch_size, 1, 1)
        if t_value.ndim == 2 and t_value.shape == (batch_size, 1):
            return t_value.reshape(batch_size, 1, 1)
        if t_value.ndim == 3 and t_value.shape == (batch_size, 1, 1):
            return t_value
        raise ValueError(f"t must be scalar or batch-shaped for batch size {batch_size}, got {t_value.shape}")


def _as_int(value: int | mx.array | None, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return int(value.item())
