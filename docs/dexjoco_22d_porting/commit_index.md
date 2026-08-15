# Commit Index

本索引按阶段记录提交意图和范围。Git commit 不能在自身内容中可靠地写入自身 hash，因此 Phase 0 的精确 hash 由包含本条目的提交对象及 `git log -- docs/dexjoco_22d_porting/` 确定；后续阶段可在新提交中回填历史 hash。

| 阶段 | 日期 | 状态 | Commit | Commit message | 范围 |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | 2026-08-15 | 本文件所在提交 | self | `docs: record DexJoCo 22d adaptation contract` | 仅新增 `docs/dexjoco_22d_porting/` 五份审计/记录文档 |

## Phase 0 提交边界

- FastWAM 起始 HEAD：`45d8e1458921d83f8ad6cf9ce993d371208dabd0`。
- DexJoCo 参考 HEAD：`8d23b0fab23b17a58c4b55f3942e17013aaf8267`，不在该提交中修改。
- 只允许 stage `docs/dexjoco_22d_porting/`。
- 提交前必须通过 `git diff --cached --check`，并人工复核 `git diff --cached --stat` 和 `git diff --cached`。
- DexJoCo 的 `environment-dexjoco.yaml` 和 `openpi/packages/openpi-client/pyproject.toml` 是用户已有修改，不属于本项目提交。
