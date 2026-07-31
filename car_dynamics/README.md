This package contains necessary modules for car dynamics modeling and model-based controller

- `model_jax`: 

    - `dbm.py`: Dynamic Bicycle Model (DBM) for vehicle dynamics modeling
    - `nn_dynamics.py`: NN Wrapper for dynamics modeling

- `controllers_jax`:
    - `mppi.py`: MPPI Implementation
    - `mppi_helper.py`: Helper functions for dealing with rollout functions

- `controllers_torch`:
    - `mppi.py`: Current Query-model MPPI shared by PyTorch and ONNX backends
    - `dbm.py`: Batched PyTorch DBM rollout used for like-for-like MPPI diagnostics
    - `alt_pure_pursuit.py`: Pure Pursuit Implementation
    - `pid.py`: PID Controller Implementation

The ROS2 `car_node` uses `controllers_torch/mppi.py`. The JAX MPPI is retained
as a legacy implementation and is not part of the current Query-model path.
