#!/usr/bin/env bash
# Wrapper for running pyslam.sif on a Mila compute node with all cache writes
# redirected away from the read-only container.
#
# Two cache tiers:
#   - Ephemeral ($SLURM_TMPDIR): JIT/codegen caches that are cheap to regenerate
#     (numba, torch_extensions, matplotlib).
#   - Persistent ($PYSLAM_MODEL_CACHE, default $SCRATCH/pyslam_cache): model
#     weights from HF Hub / torch.hub. Persists across jobs so we don't
#     re-download multi-GB checkpoints each run. Mila compute nodes have no
#     internet, so this dir must be pre-populated from a login node first.
#
# Usage:
#   scripts/run_apptainer.sh python /opt/pyslam/main_slam.py --config foo.yaml
#
# Override paths via env vars before invoking:
#   PYSLAM_SIF=/path/to/pyslam.sif
#   PYSLAM_MODEL_CACHE=$SCRATCH/pyslam_cache
#   PYSLAM_SRC=$HOME/pyslam        # bind-mount local source over baked source
#   PYSLAM_DATA=/network/datasets  # read-only data mount
#   PYSLAM_OUTPUT=$SCRATCH/runs    # writable output mount

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
: "${PYSLAM_SIF:=}"
: "${PYSLAM_MODEL_CACHE:=}"
: "${PYSLAM_SRC:=}"
: "${PYSLAM_DATA:=}"
: "${PYSLAM_OUTPUT:=}"

if [[ ! -f "$PYSLAM_SIF" ]]; then
  echo "error: SIF not found at $PYSLAM_SIF (set PYSLAM_SIF)" >&2
  exit 1
fi
if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  echo "error: \$SLURM_TMPDIR is unset — are you on a compute node?" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Pre-create cache subdirs on the writable tiers.
# ---------------------------------------------------------------------------
EPHEMERAL_CACHE="$SLURM_TMPDIR/pyslam_cache"
mkdir -p \
  "$EPHEMERAL_CACHE/numba" \
  "$EPHEMERAL_CACHE/torch_extensions" \
  "$EPHEMERAL_CACHE/matplotlib" \
  "$EPHEMERAL_CACHE/wandb" \
  "$PYSLAM_MODEL_CACHE/torch" \
  "$PYSLAM_MODEL_CACHE/huggingface"

# ---------------------------------------------------------------------------
# Env vars to pass through to the container. Apptainer only forwards vars
# prefixed with APPTAINERENV_ (and a few SLURM_* by default), so we set them
# that way here.
# ---------------------------------------------------------------------------
export APPTAINERENV_NUMBA_CACHE_DIR="$EPHEMERAL_CACHE/numba"
export APPTAINERENV_TORCH_EXTENSIONS_DIR="$EPHEMERAL_CACHE/torch_extensions"
export APPTAINERENV_MPLCONFIGDIR="$EPHEMERAL_CACHE/matplotlib"
export APPTAINERENV_WANDB_CACHE_DIR="$EPHEMERAL_CACHE/wandb"
export APPTAINERENV_WANDB_CONFIG_DIR="$EPHEMERAL_CACHE/wandb"
# Override the image-baked TORCH_HOME / HF_HOME (which point at read-only
# /opt/pyslam-cache) with the persistent scratch cache.
export APPTAINERENV_TORCH_HOME="$PYSLAM_MODEL_CACHE/torch"
export APPTAINERENV_HF_HOME="$PYSLAM_MODEL_CACHE/huggingface"
export APPTAINERENV_TRANSFORMERS_CACHE="$PYSLAM_MODEL_CACHE/huggingface"
export APPTAINERENV_HF_DATASETS_CACHE="$PYSLAM_MODEL_CACHE/huggingface/datasets"
# Force offline on compute nodes — fail fast instead of hanging on a network
# timeout if a model isn't in the cache.
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# Bind mounts.
# ---------------------------------------------------------------------------
BINDS=(
  --bind "$SLURM_TMPDIR:$SLURM_TMPDIR"
  --bind "$PYSLAM_MODEL_CACHE:$PYSLAM_MODEL_CACHE"
)
[[ -n "$PYSLAM_SRC"    ]] && BINDS+=(--bind "$PYSLAM_SRC:/opt/pyslam/pyslam")
[[ -n "$PYSLAM_DATA"   ]] && BINDS+=(--bind "$PYSLAM_DATA:/home/adam/datasets")
[[ -n "$PYSLAM_OUTPUT" ]] && BINDS+=(--bind "$PYSLAM_OUTPUT:/output")

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
exec apptainer exec --nv --nvccli --no-home "${BINDS[@]}" "$PYSLAM_SIF" "$@"
