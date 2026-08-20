#!/usr/bin/env bash
# Compile CLUEstering's CUDA backend into the analysis environment.
#
#   ./setup/build_clue_cuda.sh
#
# The conda-forge wheel ships only the CPU backends: `CLUEstering.backends` advertises
# "gpu cuda" but the .so is absent, and `run_clue` responds by printing "CUDA module not found"
# and returning with `cluster_ids` untouched rather than raising, so a timing run against the
# unbuilt backend reports a very fast no-op. scripts/bench_clue.py guards against that
# separately; this makes the backend real.
#
# Only the CLUE_GPU_CUDA target is built, so the serial and OpenMP libraries that produced every
# result in results/ are left as the wheel installed them.
#
# Needs network access: the upstream CMakeLists fetches alpaka from GitHub for its CMake target.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
# The A100 in this machine is sm_80. The upstream CMakeLists sets CUDA_ARCHITECTURES to
# "native", which asks nvcc to detect the card; pinning it makes the build reproducible and
# survives running the build on a node whose GPU is busy.
CUDA_ARCH="${CUDA_ARCH:-80}"

# The host compiler is not free to choose, for two independent reasons.
#
# First, CUDA 12.4's nvcc refuses any host gcc newer than 13, and the analysis env carries
# conda-forge gcc 14.4, so the CUDA compiler-id test fails before a line is compiled.
#
# Second, and less obviously, pybind11's type registry capsule is named after the C++ ABI
# version of the compiler that built the module, and only modules whose names match can pass
# each other's objects. The wheel's CPU modules carry cxxabi1016; building against g++-13
# (cxxabi1018) produces a module that imports cleanly, advertises "gpu cuda", and then throws
# "incompatible function arguments" the moment run_clue hands it a kernel object that
# CLUE_Convolutional_Kernels created. So the ABI tag is read off an existing module and the
# host compiler is chosen to match it rather than picked by version.
required_abi() {
  local tag
  tag=$(strings "$SITE/CLUEstering/lib/CLUE_CPU_Serial"*.so | grep -oE 'pybind11_internals_v[0-9]+_[a-z]+_[a-z]+_cxxabi[0-9]+' | head -1)
  echo "${tag##*cxxabi}"
}

pick_host_cxx() {
  local want="$1" candidate abi
  for candidate in /usr/bin/g++-13 /usr/bin/g++-12 /usr/bin/g++-11 /usr/bin/g++-9 /usr/bin/g++; do
    [ -x "$candidate" ] || continue
    abi=$(echo __GXX_ABI_VERSION | "$candidate" -E -P -x c++ - 2>/dev/null | tail -1)
    [ "$abi" = "$want" ] && { echo "$candidate"; return 0; }
  done
  return 1
}

SITE="$ENV_ANALYSIS/lib/python3.11/site-packages"
SRC="$SITE/CLUEstering/BindingModules"
BUILD="$EXTERNAL/build-clue-cuda"

[ -d "$SRC/cuda" ] || { echo "no CUDA sources at $SRC/cuda; is CLUEstering installed?" >&2; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "no nvcc at $CUDA_HOME/bin/nvcc" >&2; exit 1; }

WANT_ABI="$(required_abi)"
if [ -n "${CUDA_HOST_CXX:-}" ]; then
  echo "using host compiler $CUDA_HOST_CXX from the environment"
else
  CUDA_HOST_CXX="$(pick_host_cxx "$WANT_ABI")" || {
    echo "no installed g++ produces __GXX_ABI_VERSION=$WANT_ABI, which the wheel's CPU modules" >&2
    echo "were built with. Without a match the CUDA module cannot share pybind11's type" >&2
    echo "registry with them. Install the matching gcc, or set CUDA_HOST_CXX to override." >&2
    exit 1
  }
  echo "wheel modules need cxxabi $WANT_ABI -> host compiler $CUDA_HOST_CXX"
fi

export PATH="$CUDA_HOME/bin:$ENV_ANALYSIS/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"

# pybind11 is a build-time dependency the runtime wheel does not carry, and the major version
# MATTERS. pybind11 keeps its type registry in a capsule whose name encodes an internals
# version, and only modules sharing that version can pass each other's objects. The wheel's
# CPU modules are v4 (pybind11 2.x); building the CUDA module against pybind11 3.x produces
# v12, and `run_clue` then dies with "incompatible function arguments" the moment it hands
# gpu_cuda.mainRun a FlatKernel that CLUE_Convolutional_Kernels created. Pinned to 2.x so the
# new module joins the registry the existing ones already share.
PYBIND11_SPEC="${PYBIND11_SPEC:-pybind11<3}"
"$ENV_ANALYSIS/bin/python" - <<'PY' 2>/dev/null || "$ENV_ANALYSIS/bin/python" -m pip install --no-input "$PYBIND11_SPEC"
import pybind11, sys
sys.exit(0 if int(pybind11.__version__.split(".")[0]) < 3 else 1)
PY

PYBIND11_DIR="$("$ENV_ANALYSIS/bin/python" -c 'import pybind11; print(pybind11.get_cmake_dir())')"

# A SHIM, rather than editing the CMakeLists inside site-packages. The upstream file calls
# pybind11_add_module but never find_package(pybind11); it comments "include pybind11 extern
# subfolder" and relies on an extern/pybind11 checkout the wheel does not ship, so configuring
# it directly dies with "Unknown CMake command pybind11_add_module". The shim finds pybind11
# and then add_subdirectory()s the real thing. Because add_subdirectory keeps
# CMAKE_CURRENT_SOURCE_DIR pointing at the original location, the upstream relative include
# paths (../../../include, where the vendored CLUEstering and alpaka headers live) still
# resolve, which copying the tree somewhere else would have broken.
SHIM="$EXTERNAL/build-clue-cuda-src"
# alpaka is fetched from GitHub and is ~40 MB of headers. Kept outside the build directory so
# that wiping the build to reconfigure does not re-download it. One such re-download already
# failed mid-transfer and took the whole configure step with it.
DEPS="$EXTERNAL/clue-cuda-deps"
rm -rf "$SHIM" "$BUILD"
mkdir -p "$DEPS"
mkdir -p "$SHIM"
cat > "$SHIM/CMakeLists.txt" <<CMAKE
cmake_minimum_required(VERSION 3.16.0)
project(clue_cuda_shim LANGUAGES CXX)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
add_subdirectory($SRC bindings)
CMAKE

"$ENV_ANALYSIS/bin/cmake" -S "$SHIM" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_CXX" \
  -DCMAKE_CUDA_FLAGS="-ccbin $CUDA_HOST_CXX" \
  -DPython_EXECUTABLE="$ENV_ANALYSIS/bin/python" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DCMAKE_PREFIX_PATH="$ENV_ANALYSIS" \
  -DFETCHCONTENT_BASE_DIR="$DEPS"

# Only this target. Building "all" would also recompile the serial and OpenMP modules with a
# different toolchain than the wheel used, silently changing the baseline the CPU rows are
# measured against.
"$ENV_ANALYSIS/bin/cmake" --build "$BUILD" --target CLUE_GPU_CUDA -j "$(nproc)"

echo
echo "backends now visible to the analysis environment:"
"$ENV_ANALYSIS/bin/python" - <<'PY'
import CLUEstering
print(" ", CLUEstering.backends)
if "gpu cuda" not in CLUEstering.backends:
    raise SystemExit("CLUE_GPU_CUDA did not land in CLUEstering/lib; the build did not take")
print("  gpu cuda is live")
PY
