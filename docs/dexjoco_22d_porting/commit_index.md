# Commit Index

本索引按阶段记录提交意图和范围。Git commit 不能在自身内容中可靠地写入自身 hash，因此 Phase 0 的精确 hash 由包含本条目的提交对象及 `git log -- docs/dexjoco_22d_porting/` 确定；后续阶段可在新提交中回填历史 hash。

| 阶段 | 日期 | 状态 | Commit | Commit message | 范围 |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | 2026-08-15 | 已提交 | `835c98b8c42471a5addf5666967706957844d7f4` | `docs: record DexJoCo 22d adaptation contract` | 仅新增 `docs/dexjoco_22d_porting/` 五份审计/记录文档 |
| Phase 1 | 2026-08-15 | 已提交 | `8409c0fef83da8e88e185e8d5beb1f57b3d78131` | `feat(data): add DexJoCo 22d action pipeline` | DexJoCo v3 六任务 loader、22D/23D processor、statistics schema/CLI、smoke test 和阶段记录 |
| Phase 2 | 2026-08-15 | 已提交 | `03caff2af53632914ec7418716480b7a5ae6dbdc` | `feat(model): add dual action experts for DexJoCo` | DexJoCo Video/Arm/Hand 三 expert MoT、22D action loss、23D proprio、专用 mask/cache fail-fast 和 tiny smoke |
| Phase 3 | 2026-08-15 | 已提交 | `9854c30685985b371c890b6db7f777e50ed1e6d7` | `feat(checkpoint): add selective DexJoCo weight loading` | 精确 checkpoint key 分类、Video/Arm 加载、Action-to-Hand remap、新 projection 初始化、JSON 报告和 targeted smoke |
| Phase 4 | 2026-08-15 | 已提交 | `43f450641504ee0b0609511e0ca372a7c910f433` | `feat(train): add DexJoCo parameter groups` | DexJoCo joint post-training 冻结策略、三组 optimizer、保持 LR 比例的 scheduler 和 targeted smoke |
| Phase 5 | 2026-08-15 | 已提交 | `88c040066d6618dd24443ee0965673497df035c3` | `feat(train): close DexJoCo 22d training loop` | 六任务单步 joint training、梯度/更新审计、版本化 checkpoint artifacts、resume fail-fast 和 save/reload smoke |
| Phase 6 DexJoCo | 2026-08-15 | 已提交 | `992abca3da2bb34475485658bf76b766c49b7efa` | `feat(eval): add FastWAM DexJoCo client` | 六任务 client config、22D websocket/action adapter、23D simulator command、chunk/replan 和短 simulator smoke |
| Phase 6 DexJoCo import fix | 2026-08-15 | 已提交 | `3c85e48dc50c29e204259261eb97f3419e26e969` | `fix(eval): lazy-load DexJoCo simulator adapter` | protocol-only client 惰性加载 simulator 依赖，支持跨仓库 websocket smoke |
| Phase 6 FastWAM | 2026-08-15 | 已提交 | `34e1dd30158b05924a59496872930cf0ff01bb2a` | `feat(inference): serve DexJoCo 22d actions` | 无 cache 双 Action Expert sampler、strict websocket server、stats/T5 cache/checkpoint policy 和 targeted smoke |
| Phase 7 monitoring | 2026-08-15 | 已提交 | `98c708c7e41e48e7b31c2ccf1bb388b2e23641b0` | `feat(train): add DexJoCo TensorBoard monitoring` | rank-0 TensorBoard、loss-only validation、90/10 split、statistics parity、ZeRO-3 兼容、launcher/summary 和 4+1-step smoke |
| Phase 7 experiment | 2026-08-16 | 本文件所在提交 | self | `docs: record DexJoCo joint post-training run` | production statistics/T5 cache、20-step ZeRO-3 run hashes、loss 收敛摘要和 checkpoint 选择依据 |

## Phase 0 提交边界

- FastWAM 起始 HEAD：`45d8e1458921d83f8ad6cf9ce993d371208dabd0`。
- DexJoCo 参考 HEAD：`8d23b0fab23b17a58c4b55f3942e17013aaf8267`，不在该提交中修改。
- 只允许 stage `docs/dexjoco_22d_porting/`。
- 提交前必须通过 `git diff --cached --check`，并人工复核 `git diff --cached --stat` 和 `git diff --cached`。
- DexJoCo 的 `environment-dexjoco.yaml` 和 `openpi/packages/openpi-client/pyproject.toml` 是用户已有修改，不属于本项目提交。

## Phase 1 提交边界

- FastWAM 起始 HEAD：`835c98b8c42471a5addf5666967706957844d7f4`。
- DexJoCo 参考 HEAD：`8d23b0fab23b17a58c4b55f3942e17013aaf8267`，不在该提交中修改。
- 只 stage `configs/data/dexjoco_6task_2cam.yaml`、两个 DexJoCo scripts、DexJoCo dataset/processor/statistics 模块、两个必要的通用兼容改动、聚焦测试和本目录 Phase 1 记录。
- `/tmp` 中的官方最小下载、smoke statistics 和临时 text context 不提交。
- 提交前必须通过 `git diff --cached --check`，并人工复核 `git diff --cached --stat` 和 `git diff --cached`。

## Phase 2 提交边界

- FastWAM 起始 HEAD：`8409c0fef83da8e88e185e8d5beb1f57b3d78131`。
- DexJoCo 参考 HEAD：`8d23b0fab23b17a58c4b55f3942e17013aaf8267`，不在该提交中修改。
- 只 stage DexJoCo 专用模型/配置/smoke、`runtime.py` 新工厂和本目录 Phase 2 记录。
- 不 stage checkpoint、完整权重、训练输出、optimizer 状态或 DexJoCo 用户文件。
- 提交前必须通过 `git diff --cached --check`，并人工复核 `git diff --cached --stat` 和 `git diff --cached`。

## Phase 3 提交边界

- FastWAM 起始 HEAD：`03caff2af53632914ec7418716480b7a5ae6dbdc`。
- DexJoCo 参考 HEAD：`8d23b0fab23b17a58c4b55f3942e17013aaf8267`，不在该提交中修改。
- 只 stage DexJoCo selective checkpoint loader/config/runtime 接线、两个 targeted scripts 和本目录 Phase 3 记录。
- 不 stage checkpoint、JSON smoke 临时文件、optimizer、训练输出或 DexJoCo 用户文件。
- 提交前必须通过 `git diff --cached --check`，并人工复核 `git diff --cached --stat` 和 `git diff --cached`。

## Phase 4 提交边界

- FastWAM 起始 HEAD：`9854c30685985b371c890b6db7f777e50ed1e6d7`。
- DexJoCo 参考 HEAD：`8d23b0fab23b17a58c4b55f3942e17013aaf8267`，不在该提交中修改。
- 只 stage DexJoCo joint task config、模型冻结/分组实现、trainer optimizer/scheduler 接线、optimizer smoke 和本目录 Phase 4 记录。
- 不 stage `checkpoints/libero_uncond_2cam224.pt`、`/tmp` checkpoint 审计报告、statistics、训练输出或 DexJoCo 用户文件。
- 提交前必须通过 `git diff --cached --check`，并人工复核 `git diff --cached --stat` 和 `git diff --cached`。

## Phase 5 提交边界

- FastWAM 起始 HEAD：`43f450641504ee0b0609511e0ca372a7c910f433`。
- DexJoCo 参考 HEAD：`8d23b0fab23b17a58c4b55f3942e17013aaf8267`，不在该提交中修改。
- 只 stage trainer checkpoint/resume 契约、dual-action smoke 可观测输出、Phase 5 targeted smoke 和本目录 Phase 5 记录。
- 不 stage `checkpoints/libero_uncond_2cam224.pt`、`/tmp` 数据/statistics/checkpoint、训练输出或 DexJoCo 用户文件。
- 提交前必须通过 `git diff --cached --check`，并人工复核 `git diff --cached --stat` 和 `git diff --cached`。

## Phase 6 提交边界

- FastWAM 起始 HEAD：`88c040066d6618dd24443ee0965673497df035c3`。
- DexJoCo client commits：`992abca3da2bb34475485658bf76b766c49b7efa`、`3c85e48dc50c29e204259261eb97f3419e26e969`。
- FastWAM 只 stage inference package、DexJoCo uncached sampler、server/precompute/smoke scripts、依赖声明和本目录 Phase 6 记录。
- 不 stage checkpoint、statistics、T5 cache、simulator video、训练/评测输出或 DexJoCo 用户文件。
- 提交前必须通过 `git diff --cached --check`，并人工复核 `git diff --cached --stat` 和 `git diff --cached`。

## Phase 7 监控实现提交边界

- FastWAM 起始 HEAD：`34e1dd30158b05924a59496872930cf0ff01bb2a`。
- DexJoCo 参考 HEAD：`3c85e48dc50c29e204259261eb97f3419e26e969`；两个用户文件不修改、不 stage。
- 只 stage TensorBoard/validation/statistics split 实现、三个 Phase 7 scripts、配置/依赖和 Phase 7 项目记录。
- 不 stage event、checkpoint、statistics、T5 cache、数据、正式 run 日志或 `/tmp` smoke 产物。
- 提交前必须通过 `git diff --cached --check`，并人工复核 `git diff --cached --stat` 和完整 `git diff --cached`。

## Phase 7 实验记录提交边界

- 监控实现 commit：`98c708c7e41e48e7b31c2ccf1bb388b2e23641b0`。
- 正式成功 run 只使用该干净 commit；20 个 optimizer steps 后再提交本记录，不修改训练代码。
- 只 stage `07_tensorboard_and_training.md`、`smoke_test_ledger.md` 和 `commit_index.md`。
- 不 stage 被 `runs/` 忽略的约 81 GiB checkpoint、events、production statistics、T5 cache、日志、summary 或 run metadata。
- DexJoCo 参考 HEAD 仍为 `3c85e48dc50c29e204259261eb97f3419e26e969`；两个用户文件不修改、不 stage。
- 提交前必须通过 `git diff --cached --check`，并人工复核 `git diff --cached --stat` 和完整 `git diff --cached`。
