#!/usr/bin/env bash

# Resolve paths from this file so the setup works from any current directory.
_ANYCAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CAR_PATH="$_ANYCAR_ROOT"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "warning: activate the anycar Conda environment before sourcing set_env.sh" >&2
else
    _ANYCAR_PYTHON_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
    _ANYCAR_PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    export PYTHONPATH="$_ANYCAR_PYTHON_SITE:$CAR_PATH/car_dynamics/car_dynamics${PYTHONPATH:+:$PYTHONPATH}"

    _ANYCAR_NVRTC_LIB="$_ANYCAR_PYTHON_SITE/nvidia/cuda_nvrtc/lib"
    if [[ -d "$_ANYCAR_NVRTC_LIB" ]]; then
        export LD_LIBRARY_PATH="$_ANYCAR_NVRTC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

# Source the system ROS installation before the local workspace overlay.
if [[ -f /opt/ros/humble/setup.bash ]]; then
    source /opt/ros/humble/setup.bash
fi

if [[ -f "$CAR_PATH/install/setup.bash" ]]; then
    source "$CAR_PATH/install/setup.bash"
else
    echo "warning: $CAR_PATH/install/setup.bash not found; run colcon build first" >&2
fi

# Use the system-selected CUDA toolkit instead of a hard-coded version.
if [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH="$CUDA_HOME/bin:$PATH"
fi

# These resources are not included by the current setup.py files. Copy their
# contents idempotently into the colcon install tree when it exists.
if [[ -n "${_ANYCAR_PYTHON_VERSION:-}" ]]; then
    _ANYCAR_FOUNDATION_INSTALL="$CAR_PATH/install/car_foundation/lib/python$_ANYCAR_PYTHON_VERSION/site-packages/car_foundation"
    if [[ -d "$CAR_PATH/car_foundation/car_foundation/models" && -d "$_ANYCAR_FOUNDATION_INSTALL" ]]; then
        mkdir -p "$_ANYCAR_FOUNDATION_INSTALL/models"
        cp -a "$CAR_PATH/car_foundation/car_foundation/models/." "$_ANYCAR_FOUNDATION_INSTALL/models/"
    fi

    _ANYCAR_PLANNER_INSTALL="$CAR_PATH/install/car_planner/lib/python$_ANYCAR_PYTHON_VERSION/site-packages/car_planner"
    if [[ -d "$CAR_PATH/car_planner/assets" && -d "$_ANYCAR_PLANNER_INSTALL" ]]; then
        mkdir -p "$_ANYCAR_PLANNER_INSTALL/assets"
        cp -a "$CAR_PATH/car_planner/assets/." "$_ANYCAR_PLANNER_INSTALL/assets/"
    fi
fi

unset _ANYCAR_ROOT _ANYCAR_PYTHON_SITE _ANYCAR_PYTHON_VERSION _ANYCAR_NVRTC_LIB
unset _ANYCAR_FOUNDATION_INSTALL _ANYCAR_PLANNER_INSTALL
