# Commit Index

本索引按阶段记录提交意图和范围。Git commit 不能在自身内容中可靠地写入自身 hash，因此 Phase 0 的精确 hash 由包含本条目的提交对象及 `git log -- docs/dexjoco_22d_porting/` 确定；后续阶段可在新提交中回填历史 hash。

| 阶段 | 日期 | 状态 | Commit | Commit message | 范围 |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | 2026-08-15 | 已提交 | `835c98b8c42471a5addf5666967706957844d7f4` | `docs: record DexJoCo 22d adaptation contract` | 仅新增 `docs/dexjoco_22d_porting/` 五份审计/记录文档 |
| Phase 1 | 2026-08-15 | 已提交 | `8409c0fef83da8e88e185e8d5beb1f57b3d78131` | `feat(data): add DexJoCo 22d action pipeline` | DexJoCo v3 六任务 loader、22D/23D processor、statistics schema/CLI、smoke test 和阶段记录 |
| Phase 2 | 2026-08-15 | 本文件所在提交 | self | `feat(model): add dual action experts for DexJoCo` | DexJoCo Video/Arm/Hand 三 expert MoT、22D action loss、23D proprio、专用 mask/cache fail-fast 和 tiny smoke |

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
