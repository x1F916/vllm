# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch helpers shared by the CPU Triton kernels."""

import math
from typing import Any

from vllm.triton_utils import triton

# Triton-CPU opens an OpenMP region over the whole thread team for every
# multi-program launch. For decode-sized grids that fork/join costs more than
# the work itself, so run such grids on the calling thread instead.
SMALL_GRID_MAX_PROGRAMS = 8


def grid_num_threads(*grid: int) -> int:
    """Return the ``num_cpu_threads`` launch option for a Triton-CPU grid."""
    return 1 if math.prod(grid) <= SMALL_GRID_MAX_PROGRAMS else 0


class _LaunchState:
    def __init__(self, kernel: Any) -> None:
        from triton.runtime import driver
        from triton.runtime.jit import get_device_key

        self.binder = kernel.device_caches[get_device_key()][4]
        device = driver.active.get_current_device()
        self.stream = driver.active.get_current_stream(device)
        self.compiled: dict[tuple, Any] = {}


_STATES: dict[Any, _LaunchState] = {}


def _hook_active(hook: Any) -> bool:
    # Triton keeps launch hooks as a (possibly empty) HookChain.
    return hook is not None and bool(getattr(hook, "calls", True))


def launch(kernel: Any, grid: tuple[int, ...], *args: Any, **kwargs: Any) -> None:
    """Equivalent to ``kernel[grid](*args, **kwargs)`` with less overhead.

    Decode launches tiny kernels hundreds of times per step, and the generic
    JIT dispatch costs several microseconds each time. Specialization still
    goes through Triton's own binder, so the compiled kernel is the one
    ``kernel[grid]`` would pick; only device lookup, cache-key hashing and
    launch metadata are skipped. Falls back to the regular path when launch
    hooks are installed or on the first launch of a specialization.
    """
    knobs = triton.knobs
    if (
        kernel.pre_run_hooks
        or _hook_active(knobs.runtime.launch_enter_hook)
        or _hook_active(knobs.runtime.launch_exit_hook)
    ):
        kernel[grid](*args, **kwargs)
        return

    state = _STATES.get(kernel)
    if state is None:
        compiled = kernel[grid](*args, **kwargs)
        state = _STATES[kernel] = _LaunchState(kernel)
    else:
        compiled = None

    options = dict(
        kwargs,
        debug=kernel.debug or knobs.runtime.debug,
        instrumentation_mode=knobs.compilation.instrumentation_mode,
    )
    bound_args, specialization, _ = state.binder(*args, **options)
    key = (tuple(specialization), tuple(options.items()))
    if compiled is not None:
        state.compiled.setdefault(key, compiled)
        return
    compiled = state.compiled.get(key)
    if compiled is None:
        state.compiled[key] = kernel[grid](*args, **kwargs)
        return
    compiled.run(
        grid[0],
        grid[1] if len(grid) > 1 else 1,
        grid[2] if len(grid) > 2 else 1,
        state.stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        *bound_args.values(),
    )
