# Query 模型的 PyTorch / ONNX MPPI 接入

当前在线控制链路默认使用确定性
`TorchTransformerDecoderKinematicQueryMLP`。旧 JAX MPPI 不再由 `car_node`
调用；纯 PyTorch DBM backend 只用于同一 MPPI 配置下的诊断对照。

当前采样序列优化所使用的冻结模型、场景、状态、cost 和基准结果统一记录在
[`mppi_sampling_optimization_status_20260730.md`](mppi_sampling_optimization_status_20260730.md)。
后续 proposal 对比以该文档和对应原子快照为准。

## 固定协议

| 输入/输出 | 形状 | 语义 |
|---|---:|---|
| Raw history | `[1,250,7]` | 5 维车体系 transition + `[acceleration,steer]` |
| Initial state | `[1,5]` | `[x,y,yaw,vx,yawrate]` |
| Current action | `[1,2]` | 当前 `[acceleration,steer]`；部署图使用当前 steer |
| Candidate action | `[N,50,2]` | MPPI 候选 `[acceleration,steer]` |
| Predicted state | `[N,50,5]` | 未来 `[x,y,yaw,vx,yawrate]` |

采样周期固定为 `dt=0.05 s`，历史覆盖 12.5 秒，预测覆盖 2.5 秒。
部署图内部完成 history/context/nominal 的归一化、固定运动学 rollout、
Query residual 推理、反归一化和一致状态积分。

历史只编码一次，然后广播给全部候选 action。训练数据中第一步 context
throttle 和 `future_action[:,0,0]` 使用同一原始时间索引，因此部署图按候选
第一步 acceleration 构造 context；context steer 使用当前未 shift 的 steer。

## MPPI

`controllers_torch.mppi.TorchMPPIController` 是纯 PyTorch 实现，不依赖 JAX。
默认配置：

- 256 条候选、1 次更新；
- 8 个 action knots，线性插值为模型要求的 50 步；
- 每周期把优化序列左移一步作为 warm start；
- 固定采样标准差 `[0.25,0.35]`；
- action 范围为 `[-1,1]`；
- cost 包含 position、yaw、vx 和 action-rate。

PyTorch 和 ONNX Runtime 通过同一个四输入 rollout 接口接入 MPPI。ONNX
backend 当前使用 ONNX Runtime 的普通输入输出接口，会包含 host/device copy；
后续性能优化可以改为 I/O binding，但不会改变模型协议或 MPPI 实现。

## 导出 ONNX

```bash
source /home/plusai/miniconda3/etc/profile.d/conda.sh
conda activate anycar
source /home/plusai/anycar/set_env.sh
cd /home/plusai/anycar

python scripts/model_verify/export_query_mppi_onnx.py
```

默认产物：

```text
outputs/query_mppi/anycar_query.onnx
```

导出脚本会检查 ONNX graph，并用动态候选 batch 比较 PyTorch/ONNX 输出。

## ROS2

`car_node` 默认使用 PyTorch backend：

```bash
ros2 run car_ros2 car_node --ros-args \
  -p mppi_backend:=pytorch \
  -p mppi_num_samples:=256 \
  -p mppi_num_iterations:=1
```

使用 ONNX：

```bash
ros2 run car_ros2 car_node --ros-args \
  -p mppi_backend:=onnx \
  -p query_onnx_path:=/home/plusai/anycar/outputs/query_mppi/anycar_query.onnx
```

Quick Start 的 DBM 对照（不接回 JAX）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
  ros2 launch car_ros2 car_sim.launch.py mppi_backend:=dbm
```

可通过 `query_checkpoint` 覆盖默认实车 checkpoint。节点使用模型的
`dt=0.05` 生成当前点加未来 50 点的参考轨迹，并使用
`QueryHistoryBuffer` 按训练数据相同的车体系 transition 协议维护历史。
启动时使用当前 `vx/yawrate` 构造 constant-motion history，随后由实测状态
逐帧替换。

## 验证边界

- 当前只验证确定性均值模型，不启用 Sigma/S6-B；
- ONNX 与 PyTorch 的数值比较应在模型数据分布内进行；
- 真实车辆闭环上车前仍需核对 acceleration/steer 的执行器单位和符号；
- JAX MPPI 文件保留供历史追溯，但新节点不调用，也不要求维护一致性。

## Quick Start 场景与 DBM 对照

`car_sim.launch.py` 同时启动 `car_node` 和 numeric DBM simulator。参考线为
`car_planner/assets/cuc_inside.csv`：752 点、约 37.60 m 的闭环轨迹，范围约
10 m × 11 m，目标速度由 `GlobalTrajectory` 设置为 2.0 m/s。numeric 车辆参数为
轴距 0.21 m、质量 4.0 kg、摩擦系数 0.8。仿真步长已显式设为 0.05 s，和当前
Query/MPPI 协议一致。

2026-07-30 使用相同 seed、256 samples、50 步 horizon 和 cost，分别只替换
PyTorch Query/DBM rollout。两组都从零状态开始，按共同前 189 步（9.45 s）统计：

| 指标 | Query PyTorch | DBM PyTorch |
|---|---:|---:|
| 横向误差 MAE | 0.435 m | 0.113 m |
| 横向误差 RMSE | 0.609 m | 0.142 m |
| 横向误差 P95 | 1.319 m | 0.286 m |
| 纵向速度误差 MAE | 0.900 m/s | 0.337 m/s |
| MPPI 平均计算时间 | 29.0 ms | 66.1 ms |
| 50 ms deadline miss | 0.53% | 100% |

因此这组小车仿真中 Query 横向 MAE 是 DBM 的 3.85 倍，但推理平均快约
37.1 ms。该结果不能解释为 Query 架构普遍弱于 DBM：当前 checkpoint 的
轴距为 3.9 m、训练 context 速度均值为 16.56 m/s，而 Quick Start 是轴距
0.21 m、目标速度 2.0 m/s 的小车，明显处于训练分布外。若要把 Query 用于此
场景，应使用 small-car 数据和对应几何参数训练/微调 checkpoint。

原始 bag、逐步数据和 JSON 汇总位于
`outputs/mppi_dbm_query_comparison/20260730/`。复算命令：

```bash
python scripts/model_verify/compare_mppi_rosbags.py \
  --query-bag outputs/mppi_dbm_query_comparison/20260730/query_pytorch_dt005_v2 \
  --dbm-bag outputs/mppi_dbm_query_comparison/20260730/dbm_torch_dt005 \
  --output-dir outputs/mppi_dbm_query_comparison/20260730/dt005_comparison \
  --sim-dt 0.05 --model-dt 0.05
```
