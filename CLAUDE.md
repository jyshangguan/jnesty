# JNesty Development Notes for Claude

This file tells Claude Code how to work with me on this repository.

## Project

This repository develops **JNesty**, a JAX-based nested sampling package with GPU acceleration and a Dynesty-like user interface.

Repository path:

```text
/home/shangguan/Softwares/my_modules/JNesty
```

Main development target:

```text
src/jnesty/
```

Important files:

```text
src/jnesty/__init__.py            # Public imports
src/jnesty/jnesty.py              # Public NestedSampler API (thin wrapper)
src/jnesty/sampler.py             # Core NS loop (delegates to Bound + InternalSampler)
src/jnesty/internal_samplers.py   # Proposal strategies (RWalkSampler, future: SliceSampler)
src/jnesty/bounding.py            # Bound base class + UnitCube, SingleEllipsoid, MultiEllipsoid
src/jnesty/multi_ellipsoid.py     # JIT-compiled multi-ellipsoid fitting core
src/jnesty/results.py             # Results class + format_results()
src/jnesty/utils.py               # Shared utilities (randsphere, logsubexp, etc.)
src/jnesty/plotting.py            # Dynesty-style plotting helpers
```

Architecture follows Dynesty's separation of concerns:

```text
NestedSampler (jnesty.py)            # User-facing API
  └─> run_nested_sampling (sampler.py)  # Core NS loop (two-phase)
        │   Phase 1: uniform rejection sampling (sampler.py)
        │   Phase 2: random walk with adaptive bounds (sampler.py)
        ├─> Bound (bounding.py)           # Pluggable bounding methods
        └─> InternalSampler (internal_samplers.py)  # Pluggable proposal strategies
  └─> format_results (results.py)        # Raw result -> Results object
```

Sampling modes:

- **Legacy mode** (`queue_size=0`): batch parallel walks, each with `rwalk_K // batch_size` steps
- **Queue mode** (`queue_size>1`, default for `bound='multi'`): Dynesty-style GPU queue where each entry gets full `rwalk_K` steps; scale adapts at queue drain

Demo and development files:

```text
dev/demo/                                  # Demo scripts and examples
dev/problems.md                            # Important unresolved problems
dev/develop_log.md                         # High-level development notes only
dev/task_*/                                # Optional task-specific working folders
```

## Main goals

The package should provide:

- A simple nested sampling API similar to Dynesty.
- GPU-accelerated likelihood evaluation using JAX.
- Correct convergence behavior with true early stopping.
- Useful diagnostics and plotting tools.
- Clear, maintainable, testable code.

The current public-facing API is centered on:

```python
from jnesty import NestedSampler, plotting
from jnesty import save_results, load_results
```

## Collaboration style

### Keep reports short

Do not produce long reports unless I explicitly ask for details.

For routine work, use this format:

```text
Result: one-sentence summary.
Key points:
- Point 1
- Point 2
Checks: command and pass/fail status.
Next: only if user action is needed.
```

For most successful routine tasks, keep the final report within 5-8 lines.

Do not include:

- Long logs.
- Full command transcripts.
- Repeated background explanations.
- Long lists of every small edit.
- Verbose implementation narratives.

For failed checks, report only:

```text
Failed command: ...
Core error: ...
Likely cause: ...
Proposed fix: ...
```

Only include long logs when I ask for them.

### Reduce unnecessary interaction

Minimize interruptions for trivial work.

Do not ask for confirmation before:

- Reading files in the project.
- Searching within the project.
- Running simple read-only checks.
- Running syntax checks.
- Running import checks.
- Running project-local tests.
- Making small, clearly implied code edits.

Ask me before:

- Deleting files.
- Moving many files.
- Rewriting git history.
- Running `git clean`, `git reset --hard`, or force-push commands.
- Installing packages.
- Using the network.
- Changing public APIs.
- Making broad refactors.
- Changing numerical algorithms in a way that may affect scientific results.
- Running long computations.
- Editing files outside the repository.

When the intent is clear, make reasonable implementation decisions without asking about trivial details.

## Command and permission rules

The goal is to avoid unnecessary Claude Code permission prompts during safe checks.

### Prefer simple explicit commands

Use explicit paths and literal arguments.

Good:

```bash
cd /home/shangguan/Softwares/my_modules/JNesty
python dev/demo/test_plotting.py
python -m py_compile src/jnesty/api.py
git status
git diff --stat
git log --oneline -5
```

Avoid shell expansions in simple checks:

```bash
$VAR
${VAR}
~
*.py
{a,b}
$(...)
`...`
```

Avoid commands like:

```bash
cd ~/Softwares/my_modules/JNesty
PYTHONPATH=$PWD/src:$PYTHONPATH python dev/demo/test_plotting.py
python *.py
```

### Avoid problematic `python -c`

Do not propose `python -c` commands containing:

- Comments.
- Multi-line code.
- Heredocs.
- Complex shell quoting.
- Environment variable expansion.

Bad:

```bash
python -c "
# check imports
import jnesty
print(jnesty.__file__)
"
```

Prefer existing scripts/tests. If a Python check needs multiple lines, create a small temporary script inside the project tree, run it explicitly, and remove it afterward if it is not useful.

Good:

```bash
python dev/check_imports.py
```

### Batch trivial checks

When several simple checks are needed, group them into one or two safe commands instead of repeatedly asking for permission.

Prefer one meaningful test command over many tiny exploratory commands.

## Development workflow

### Start by inspecting the relevant code

Before editing, inspect the relevant files and understand the current implementation.

Do not assume the architecture from memory if the answer can be checked quickly in the repository.

### Make focused changes

Prefer small, focused edits that solve the current problem.

Avoid broad rewrites unless clearly necessary.

Preserve the existing public API unless changing it is the explicit task.

### Test incrementally

After code changes, run the smallest meaningful check first.

Examples:

```bash
python -m py_compile src/jnesty/sampler.py
python -m py_compile src/jnesty/internal_samplers.py
python -m py_compile src/jnesty/jnesty.py
pytest tests/integration/test_queue_and_defaults.py
```

If sampler behavior changes, run at least one representative demo.

If plotting changes, run a plotting-specific test or demo.

If public API changes, test import behavior.

### Keep temporary files local

Temporary files should stay inside the repository, preferably under:

```text
dev/tmp/
```

Remove temporary files when they are no longer useful.

## Task notes

For major tasks, create a folder:

```text
dev/task_XXX_short_name/
```

Useful files inside a task folder:

```text
plan.md         # Short implementation plan
develop_log.md  # Only important progress and decisions
README.md       # Final task summary if useful
```

Keep these files concise. They should help future development, not record every command.

A good `plan.md` is short:

```markdown
# Plan: Feature name

## Goal
One paragraph.

## Steps
- Step 1
- Step 2
- Step 3

## Checks
- Check 1
- Check 2
```

A good `develop_log.md` records only durable information:

```markdown
# Development Log

## YYYY-MM-DD

- Implemented ...
- Fixed ...
- Important decision: ...
- Remaining issue: ...
```

## Core technical principles

### Critical Development Principle: Check Dynesty First

**When investigating sampling algorithm problems:**

ALWAYS investigate how Dynesty handles the issue before implementing custom solutions.

**Process:**
1. Reproduce the problem with Dynesty
2. Examine Dynesty's source code to understand their approach
3. Check Dynesty's configuration and default parameters
4. Only implement custom solutions if Dynesty's approach doesn't apply

**Why this matters:**
- Dynesty has years of battle-tested optimization
- Many nested sampling problems have known solutions
- Custom approaches often introduce subtle bugs
- Dynesty's defaults are tuned for real-world problems

**Common areas to check Dynesty first:**
- Proposal scale adaptation strategies
- Bounding methods (ellipsoid, multi-ellipsoid, none)
- Sampling efficiency tuning
- Convergence criteria implementation
- Live point management

### JAX control flow

Use JAX-compatible control flow inside JIT-compiled code.

Important distinction:

- `lax.while_loop` can terminate based on a condition.
- `lax.fori_loop` always runs a fixed number of iterations.

For nested sampling convergence, prefer `lax.while_loop` when early stopping is required.

### Two-phase sampling

The sampler uses a two-phase approach matching Dynesty:

1. **Phase 1 (uniform rejection)**: Draws batches of uniform random points from the unit cube, picks the first valid replacement (logL > worst). Continues until efficiency drops below `min_eff` (default 10%) or `min_ncall` (default 2*nlive) calls are reached. JIT-compiled via `lax.while_loop`.

2. **Phase 2 (random walk)**: Uses `RWalkSampler` with adaptive bounds. Runs in either:
   - **Legacy mode**: batch parallel walks with `batch_size` walks of `rwalk_K // batch_size` steps each
   - **Queue mode** (default for `bound='multi'`): Dynesty-style GPU queue where each entry gets full `rwalk_K` steps; scale adapts only at queue drain

The bound is fitted from live points at the Phase 1 → Phase 2 transition.

### Internal loop state tuple (Phase 2)

The core loop packs mutable state into a flat tuple for `lax.while_loop`. Base state is 18 elements; queue mode extends to 23:

| Idx | Variable | Description |
|-----|----------|-------------|
| 0 | live_x | (nlive, ndim) live points |
| 1 | live_logL | (nlive,) live logL |
| 2-5 | buffers | dead point x, logL, delta_logZ, scale trajectories |
| 6 | logZ | running evidence |
| 7 | delta_logZ | convergence metric |
| 8 | iteration | loop counter |
| 9 | key | PRNG key |
| 10 | scale | current proposal scale |
| 11 | hist_accept | accumulated acceptances (reset at bound update) |
| 12 | hist_total | accumulated total proposals (reset at bound update) |
| 13 | bound_axes | (ndim, ndim) axes from bound |
| 14 | me_axes | (max_ell, ndim, ndim) multi-ellipsoid axes |
| 15 | me_logvol_ells | (max_ell,) multi-ellipsoid volumes |
| 16 | calls_at_update | total_calls at last bound update |
| 17 | total_calls | total likelihood calls (non-resetting) |
| 18-22 | queue arrays | (queue mode only) queue_x, queue_logL, queue_nacc, queue_ntot, queue_head |

### Convergence

The main convergence condition is based on:

```text
delta_logZ < threshold
```

Here `delta_logZ` estimates the remaining evidence contribution from live points.

### Results format

Results should remain close to Dynesty-style dictionary output when practical.

Important fields include:

```python
results = {
    'logz': float,
    'logzerr': float,
    'information': float,
    'samples': ndarray,            # (N, ndim) dead + live points (physical)
    'samples_u': ndarray,          # (N, ndim) dead + live points (unit cube)
    'logl': ndarray,               # (N,) log-likelihoods
    'logwt': ndarray,              # (N,) log importance weights
    'logvol': ndarray,             # (N,) log prior volumes
    'logz_trajectory': ndarray,    # (N,) cumulative evidence
    'logzerr_trajectory': ndarray, # (N,) evidence error trajectory
    'delta_logZ_trajectory': ndarray,  # (niter,) convergence history
    'scale_trajectory': ndarray,   # (niter,) proposal scale history
    'nlive': int,
    'niter': int,
    'eff': float,
    'acceptance_rate': float,
    'converged': bool,
    'delta_logz': float,
    'delta_logZ_threshold': float,
    'rwalk_K': int,
}
```

### DeviceArray handling

Convert JAX arrays to NumPy before plotting with Matplotlib.

Use a helper like:

```python
def _convert_to_numpy(arr):
    if hasattr(arr, '__array__'):
        return np.array(arr)
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        return np.asarray(arr)
```

### Progress bar

Progress tracking should not dominate runtime.

If using `io_callback` with `tqdm`, throttle updates rather than updating every iteration.

A typical update interval is every 100 iterations.

### Dimension-aware defaults

Reasonable current defaults:

```python
rwalk_K = max(25, ndim + 20)
rwalk_step_scale = 1.0  # Dynesty default
batch_size = max(1, rwalk_K // max(1, rwalk_K * 10 // nlive))  # ~5 steps/walk
queue_size = 8  # for bound='multi', Dynesty-style GPU parallelism
bound_update_interval = rwalk_K * nlive  # in likelihood calls, matching Dynesty
```

Change these only with tests or clear justification.

## Code style

### Imports

Use this general order:

```python
# Standard library
import os
import sys

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# JAX
import jax
import jax.numpy as jnp
from jax import random

# Local package
from jnesty import NestedSampler, plotting
```

### Docstrings

Use NumPy-style docstrings for public functions and classes.

Keep internal comments useful and concise.

## Testing guide

Before committing or reporting completion, choose tests appropriate to the change.

### Small code edit

```bash
python -m py_compile path/to/changed_file.py
```

### API change

Run an import check and a minimal API example.

### Sampler behavior change

Run at least one representative demo and compare output qualitatively with expectations.

Useful demos:

```text
dev/demo/
```

### Plotting change

Run the plotting test or plotting demo.

```bash
python dev/demo/test_plotting.py
```

## Git rules

Safe commands can be used freely:

```bash
git status
git diff --stat
git diff
git log --oneline -10
```

Ask before:

```bash
git reset --hard
git clean -fdx
git rebase
git push --force
git checkout -- path/to/file
```

Do not discard user changes without explicit approval.

Commit only when requested or when the workflow clearly calls for it.

Use concise commit messages.

## What not to do

Do not:

- Produce long reports for routine tasks.
- Ask for permission on trivial read-only checks.
- Use shell expansions in simple commands when explicit paths work.
- Use multi-line commented `python -c` commands.
- Make broad refactors without need.
- Rewrite public APIs casually.
- Hide numerical changes in unrelated edits.
- Delete or overwrite user work without explicit approval.
- Record excessive command-by-command logs in markdown files.
