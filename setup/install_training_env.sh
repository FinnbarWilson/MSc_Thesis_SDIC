#!/usr/bin/env bash
# Build the TRAINING environment: torch + hepattn + the thesis patch. Needs a GPU.
#
#   ./setup/install_training_env.sh
#
# The scripted form of the three steps in src/maskformer/README.md: clone hepattn, apply
# hepattn-changes.patch, copy hepattn_colliderml/ over the experiment directory. hepattn is
# cloned rather than vendored, which is an authorship decision that README explains. Everything
# lands in external/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/paths.sh"

# cb4fb10 "Add ColliderML Experiment" is the surviving upstream commit introducing the
# experiment this work uses, and hepattn-changes.patch applies to it cleanly. An earlier
# reference commit was rebased away upstream, so vendor the tree if the
# original checkpoint's provenance ever matters more than convenience.
COMMIT="${HEPATTN_COMMIT:-cb4fb10}"

echo "=== [1/5] venv at $VENV_TRAIN (python 3.12, which hepattn pins with ==) ==="
mkdir -p "$EXTERNAL"
python3.12 -m venv "$VENV_TRAIN"
"$VENV_TRAIN/bin/pip" install -q --upgrade pip wheel setuptools

echo "=== [2/5] torch ==="
# cu128 + torch 2.9 matches the flash-attn wheel hepattn pins. The driver here is 550.54.14
# (CUDA 12.4); CUDA minor-version compatibility covers a 12.8 runtime on a >=525 driver, and the
# check below verifies that on the actual card rather than assuming it.
"$VENV_TRAIN/bin/pip" install -q torch==2.9.* --index-url https://download.pytorch.org/whl/cu128
"$VENV_TRAIN/bin/pip" install -q numpy   # torch warns "Failed to initialize NumPy" without it

if ! "$VENV_TRAIN/bin/python" - <<'PY'
import sys, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("CUDA not available")
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("bf16 matmul ok:", float((x @ x).float().abs().mean()))
free, total = torch.cuda.mem_get_info()
print(f"GPU memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
PY
then
    echo "!!! cu128 torch failed against this driver; falling back to cu126"
    "$VENV_TRAIN/bin/pip" install -q --force-reinstall torch==2.9.* --index-url https://download.pytorch.org/whl/cu126
    "$VENV_TRAIN/bin/python" -c "import torch;assert torch.cuda.is_available();print('cu126 fallback OK')"
fi

echo "=== [3/5] hepattn at $COMMIT + the thesis patch ==="
[ -d "$HEPATTN/.git" ] || git clone -q https://github.com/samvanstroud/hepattn "$HEPATTN"
git -C "$HEPATTN" fetch -q --all
git -C "$HEPATTN" checkout -q -- .
git -C "$HEPATTN" checkout -q "$COMMIT"
if git -C "$HEPATTN" apply --reverse --check "$REPO_ROOT/src/maskformer/hepattn-changes.patch" 2>/dev/null; then
    echo "patch already applied"
else
    git -C "$HEPATTN" apply "$REPO_ROOT/src/maskformer/hepattn-changes.patch"
    echo "patch applied (weighted DICE loss, constituent_weight_field, writer output_name)"
fi

echo "=== [4/5] the mirrored colliderml subtree ==="
# src/maskformer/hepattn_colliderml/ is the source of record for these files; verify_sync.sh
# checks the two stay identical.
DEST="$HEPATTN/src/hepattn/experiments/colliderml"
mkdir -p "$DEST"/{configs,eval}
cp "$REPO_ROOT/src/maskformer/hepattn_colliderml/"*.py          "$DEST/"
cp "$REPO_ROOT/src/maskformer/hepattn_colliderml/configs/"*.yaml "$DEST/configs/"
cp "$REPO_ROOT/src/maskformer/hepattn_colliderml/eval/"*.py      "$DEST/eval/"
echo "copied $(ls "$REPO_ROOT/src/maskformer/hepattn_colliderml/"*.py | wc -l) modules + configs into the checkout"

echo "=== [5/5] hepattn and its dependencies ==="
"$VENV_TRAIN/bin/pip" install -q cmake ninja pybind11 scikit-build-core
# --ignore-requires-python is not optional here, and it is not papering over a real conflict.
# hepattn's pyproject declares `requires-python = "== 3.12"`. Under PEP 440 `== 3.12` means
# exactly 3.12.0, not the 3.12 series; expressing "any 3.12" needs `== 3.12.*`. So pip refuses
# every patch release, including the 3.12.3 that Ubuntu 24.04 ships:
#     ERROR: Package 'hepattn' requires a different Python: 3.12.3 not in '==3.12'
# Upstream does not hit this because pixi resolves the interpreter itself rather than going
# through pip's check. The interpreter genuinely is 3.12, which is what hepattn means.
"$VENV_TRAIN/bin/pip" install -e "$HEPATTN" --no-build-isolation --ignore-requires-python 2>&1 | tail -5
# flash-attn is required, not optional: both configs set attn_type: flash-varlen.
"$VENV_TRAIN/bin/pip" install -q "flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.17/flash_attn-2.8.3%2Bcu128torch2.9-cp312-cp312-linux_x86_64.whl" \
    || echo "!!! flash-attn wheel failed; attn_type: flash-varlen will not run"

echo
echo "=== summary ==="
"$VENV_TRAIN/bin/python" - <<'PY'
for m in ["torch","lightning","comet_ml","awkward","hepattn","flash_attn","lion_pytorch","jsonargparse"]:
    try:
        __import__(m); print(f"  OK      {m}")
    except Exception as e:
        print(f"  MISSING {m}  ({type(e).__name__})")
PY
echo "training env: $VENV_TRAIN"
