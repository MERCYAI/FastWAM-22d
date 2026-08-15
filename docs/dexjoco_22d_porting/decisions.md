# 技术决策记录

## D-0001：冻结 DexJoCo 单臂 22D 动作契约

- 日期：2026-08-15
- 状态：Accepted
- 决策：`action[..., :6]` 是绝对 TCP `xyz + rotvec`，`action[..., 6:22]` 是 Allegro Hand 16D。
- 证据：`scripts/record_demos_zarr.py:224-254` 的 quaternion-to-rotvec 写入；`dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py:204-225` 的逆转换；`dexjoco/dexjoco/tasks/policy_wrappers.py:5-16` 的 simulator action 定义；六任务 episode 0 的 22D float32 schema。
- 影响：模型内部 action chunk 使用 `[B,T,22]`，环境单步消费 `[22]`。若后续证据冲突，停止并报告，不能静默重排。

## D-0002：冻结 23D state 排列和 quaternion 顺序

- 日期：2026-08-15
- 状态：Accepted
- 决策：`state = xyz(3) + quat_wxyz(4) + hand(16)`。
- 证据：六任务 `proprio_keys` 的 `tcp_pose, gripper_pose` 起始顺序，`DexjocoObsAdapter.observation` 的按 key flatten，rand_obj/rand_full converter 的前 23D slice，以及 `DexJoCoOpenPIEnv.stay` 对 state quaternion 的 SciPy `scalar_first=True` 读取。
- 影响：后续 proprio encoder 输入维必须是 23；所有 quaternion adapter 必须显式声明顺序。

## D-0003：维度相关 projection 全部新初始化

- 日期：2026-08-15
- 状态：Accepted
- 决策：Arm 和 Hand 的 `action_encoder/head`、23D proprio encoder 都 newly initialized；不得截取或部分复制旧 7D projection。
- 理由：旧 7D projection 的特征坐标不对应新的 Arm 6D/Hand 16D 或 22D 联合语义，机械复制会制造无证据的参数含义。
- 影响：后续 checkpoint loader 应报告这些 key 为有意新建，而不是把 shape mismatch 隐藏为兼容加载。

## D-0004：backbone 权重迁移按组件分开

- 日期：2026-08-15
- 状态：Accepted
- 决策：Video DiT backbone pretrained；Arm Transformer backbone pretrained；Hand Transformer backbone 从旧 Action Expert remap。
- 约束：当前仓库只有一个 `ActionDiT`，因此 Arm/Hand 是后续目标结构。remap 必须有显式 key 映射、shape 检查和加载报告。

## D-0005：Phase 0 采用最小只读数据抽样

- 日期：2026-08-15
- 状态：Accepted
- 决策：每任务只读取 episode 0 的低维数据和必要 metadata；不扫描全数据集。
- 限制：没有可用的 MP4 解码验证，因此相机 shape/FPS/codec 是 metadata 结论，时间对齐是 metadata 加 converter 锁步代码结论。
- 影响：像素内容、丢帧和完整数据统计仍需后续 smoke test。

## D-0006：以真实运行入口描述 server 能力

- 日期：2026-08-15
- 状态：Accepted
- 决策：FastWAM 当前记录为本地 runtime、LIBERO eval 和 RoboTwin deploy；不宣称存在通用模型 server。DexJoCo 的现有网络 client 是 OpenPI `WebsocketClientPolicy`。
- 影响：后续集成需明确新增 adapter 还是复用已有服务协议。

## D-0007：Phase 0 的 Git 范围仅限文档目录

- 日期：2026-08-15
- 状态：Accepted
- 决策：只 stage `docs/dexjoco_22d_porting/` 下五个文件。
- 约束：不修改或 stage DexJoCo 的 `environment-dexjoco.yaml` 和 `openpi/packages/openpi-client/pyproject.toml`；保留所有用户改动。
