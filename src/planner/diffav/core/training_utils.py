"""Shared training utilities for gradient safety and diagnostics.

Provides JIT-compatible gradient processing functions used by both
the trajectory trainer and the DPO alignment trainer.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def nan_safe_gradients(
    loss: jax.Array,
    grads: Any,
) -> tuple[Any, jax.Array]:
    """Zero gradients when NaN is detected in loss or gradient leaves.

    JIT-compatible: uses ``jnp.where`` instead of Python branching.

    Args:
        loss: Scalar loss value to check for NaN.
        grads: PyTree of gradients (e.g. from ``nnx.value_and_grad``).

    Returns:
        Tuple of ``(safe_grads, has_nan)`` where ``safe_grads`` has all
        array leaves zeroed if any NaN was found, and ``has_nan`` is a
        scalar boolean array.
    """
    grad_leaves = jax.tree_util.tree_leaves(grads)

    has_nan = jnp.isnan(loss)
    for g in grad_leaves:
        if hasattr(g, "shape"):
            has_nan = has_nan | jnp.any(jnp.isnan(g))

    safe_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(has_nan, jnp.zeros_like(g), g) if hasattr(g, "shape") else g,
        grads,
    )
    return safe_grads, has_nan
