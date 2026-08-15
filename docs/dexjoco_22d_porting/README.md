# FastWAM x DexJoCo 22D 迁移记录

## 项目目标

本目录记录 FastWAM 适配 DexJoCo 单臂 22D 动作空间的证据、契约、决策、验证结果和提交索引。目标是在不混淆现有 7D Action Expert 权重语义的前提下，逐阶段完成 DexJoCo 六个任务的数据接入、模型适配、训练和仿真评测。

Phase 0 只做仓库审计、轻量数据抽样、动作/状态契约冻结和记录初始化；不修改模型、数据、训练或推理代码，也不运行训练、推理或仿真闭环。

Phase 1 增加 DexJoCo LeRobot v3 六任务数据路径、22D/23D processor 和 versioned normalization statistics；不修改模型结构或训练/推理算法。

Phase 2 增加 DexJoCo 专用 Video/Arm/Hand 三 expert MoT 结构、23D proprio encoder 接线和完整 22D action loss；不实现 checkpoint 迁移、optimizer 分组或正式推理。

## 任务范围

项目覆盖以下六个 DexJoCo 任务：

1. `water_plant`
2. `hammer_nail`
3. `click_mouse`
4. `pick_bucket`
5. `pinch_tongs`
6. `fold_glasses`

## 仓库基线

| 仓库 | 真实绝对路径 | 初始分支 | 初始 HEAD |
| --- | --- | --- | --- |
| FastWAM | `/home/user/fastwam-22d` | `main` | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| DexJoCo | `/home/shared/ai/datasets/DexJoCo/dexjoco` | `main` | `8d23b0fab23b17a58c4b55f3942e17013aaf8267` |

用户提供的 `/home/shared/ai/datasets/DexJoCo` 是容器目录，不是 Git 根目录。实际 LeRobot 数据根目录为 `/home/shared/ai/datasets/dexlewm/dexjoco`；它不是本项目的代码提交目标。

Phase 0 开始时 FastWAM 工作树干净。DexJoCo 工作树已有以下用户未提交修改，所有阶段均须保留，除非用户明确授权处理：

- `environment-dexjoco.yaml`
- `openpi/packages/openpi-client/pyproject.toml`

## 记录文件

- `00_audit_and_contract.md`：Phase 0 的仓库、代码、数据和模拟器审计，以及冻结的 22D 正式契约。
- `01_data_and_normalization.md`：Phase 1 的 v3 数据入口、batch 契约、normalization schema、命令、smoke 结果和风险。
- `02_dual_action_experts.md`：Phase 2 的双 Action Expert 结构、联合 attention、loss、cache 边界和 tiny forward smoke。
- `decisions.md`：跨阶段技术决策及其证据、影响和变更条件。
- `smoke_test_ledger.md`：按阶段记录轻量检查、smoke test、未运行项和结果证据。
- `commit_index.md`：按阶段记录提交目的、范围和 commit message；提交 hash 由该条目所在 Git 历史确定。

## 每阶段记录规范

每个阶段至少记录以下项目：

- 日期。
- 阶段目标和明确边界。
- 阶段开始时两个仓库的分支、HEAD 和 `git status --short`。
- 技术决策及其代码或数据证据。
- 修改文件清单。
- 实际执行的关键命令或可复现的命令类别。
- smoke test 结果，以及明确未运行的验证。
- 已知风险、待确认项和停止条件。
- 进入后续阶段的前置条件。
- 计划使用的 commit message。

记录必须区分已由真实代码确认、已由样本确认、仅由 metadata 确认和待确认四种结论。不得仅凭张量维度推断语义，不得把未执行的训练、推理、视频解码或仿真验证写成通过。

## Phase 0 冻结契约摘要

```text
action[..., :6]    = absolute TCP xyz(3) + rotvec(3)
action[..., 6:22]  = Allegro Hand 16D
state               = xyz(3) + quat wxyz(4) + hand(16) = 23D
```

完整证据、执行器顺序和默认权重迁移策略见 `00_audit_and_contract.md`。若后续代码或数据证据与该契约冲突，必须停止并报告，不能静默改变契约。
