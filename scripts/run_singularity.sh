#!/usr/bin/env bash
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
# Detect runtime: apptainer or singularity
# ---------------------------------------------------------------------------
if command -v apptainer &>/dev/null; then
  RUNTIME=apptainer
  ENVPREFIX=APPTAINERENV
  NV_FLAGS=(--nv --nvccli)
elif command -v singularity &>/dev/null; then
  RUNTIME=singularity
  ENVPREFIX=SINGULARITYENV
  #NV_FLAGS=(--nv)
else
  echo "error: neither apptainer nor singularity found in PATH" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Cache dirs
# ---------------------------------------------------------------------------
EPHEMERAL_CACHE="$SLURM_TMPDIR/pyslam_cache"
CONTAINER_HOME="$PYSLAM_MODEL_CACHE/home"
mkdir -p \
  "$EPHEMERAL_CACHE/numba" \
  "$EPHEMERAL_CACHE/torch_extensions" \
  "$EPHEMERAL_CACHE/matplotlib" \
  "$EPHEMERAL_CACHE/wandb" \
  "$PYSLAM_MODEL_CACHE/torch" \
  "$PYSLAM_MODEL_CACHE/huggingface" \
  "$CONTAINER_HOME"

# ---------------------------------------------------------------------------
# Env vars (runtime-agnostic)
# ---------------------------------------------------------------------------
export ${ENVPREFIX}_XDG_CACHE_HOME="$CONTAINER_HOME/.cache"
export ${ENVPREFIX}_XDG_CONFIG_HOME="$CONTAINER_HOME/.config"
export ${ENVPREFIX}_NUMBA_CACHE_DIR="$EPHEMERAL_CACHE/numba"
export ${ENVPREFIX}_TORCH_EXTENSIONS_DIR="$EPHEMERAL_CACHE/torch_extensions"
export ${ENVPREFIX}_MPLCONFIGDIR="$EPHEMERAL_CACHE/matplotlib"
export ${ENVPREFIX}_WANDB_CACHE_DIR="$EPHEMERAL_CACHE/wandb"
export ${ENVPREFIX}_WANDB_CONFIG_DIR="$EPHEMERAL_CACHE/wandb"
export ${ENVPREFIX}_TORCH_HOME="$PYSLAM_MODEL_CACHE/torch"
export ${ENVPREFIX}_HF_HOME="$PYSLAM_MODEL_CACHE/huggingface"
export ${ENVPREFIX}_TRANSFORMERS_CACHE="$PYSLAM_MODEL_CACHE/huggingface"
export ${ENVPREFIX}_HF_DATASETS_CACHE="$PYSLAM_MODEL_CACHE/huggingface/datasets"
export ${ENVPREFIX}_HF_HUB_OFFLINE=1
export ${ENVPREFIX}_TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# Bind mounts
# ---------------------------------------------------------------------------
BINDS=(
  --bind "$SLURM_TMPDIR:$SLURM_TMPDIR"
  --bind "$PYSLAM_MODEL_CACHE:$PYSLAM_MODEL_CACHE"
  --bind "$HOME/repos/pyslam/data:/opt/pyslam/data"
  --bind "$HOME/repos/pyslam/main_slam.py:/opt/pyslam/main_slam.py"
  --bind "$HOME/repos/pyslam/config.yaml:/opt/pyslam/config.yaml"
  --bind "$HOME/repos/pyslam/pyslam/config_parameters.py:/opt/pyslam/pyslam/config_parameters.py"
)
[[ -n "$PYSLAM_SRC"    ]] && BINDS+=(--bind "$PYSLAM_SRC:/opt/pyslam/pyslam")
[[ -n "$PYSLAM_DATA"   ]] && BINDS+=(--bind "$PYSLAM_DATA:/home/adam/datasets")
[[ -n "$PYSLAM_OUTPUT" ]] && BINDS+=(--bind "$PYSLAM_OUTPUT:/output")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
exec "$RUNTIME" exec "${NV_FLAGS[@]}" \
  --home "$CONTAINER_HOME:$CONTAINER_HOME" \
  "${BINDS[@]}" "$PYSLAM_SIF" "$@"
