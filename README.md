# Work2 论文复现源码

拟投稿期刊：[Information Sciences](https://www.sciencedirect.com/journal/information-sciences)。

本仓库合并 `WORK2_REPRO_CODE_ONLY.zip` 与 `WORK2_VEST2_A4_SOURCE_AND_S6_ENTRY.zip`，按 SHA256 去重，保留论文方法、运行入口及必要调用依赖。实验沿用既有服务器环境和冻结输入；本次交付只整理、上传源码，不重新运行实验。

## 方法与入口

| 方法 | 仓库中的核心文件 | 原始入口 |
|---|---|---|
| RCPS | `frozen/representation/audit_bcr_a4_rcps.py` | `build_synth_library()`、`choose_bcct()` |
| TECS | `frozen/representation/audit_tree_embeddable_certificate_synthesis_v0.py` | `build_tecs_library()` |
| RBJS | `frozen/representation/audit_repairability_bounded_joint_search_v0.py` | `bounded_choose_bcct()` |
| APQC | `frozen/representation/audit_answer_pattern_quotient_compiler_v0.py` | `apqc_build()` |
| VEST-2 | 与 RCPS 共用的 pinned A4 文件，保留一份 | `choose_bcct()` 第 617 行；`choose_second()` 第 388 行 |
| S6 比较入口 | `frozen/evaluation/dl85_comparison_driver.py` | 第 74 行调用 `mod.choose_bcct(lib, valid)[0]` |
| RPC | `frozen/planning/frozen_worker.py` 与同目录边界模块 | 原始 exact worker；`rpc_train82_driver.py` 为三轮入口 |
| 联合精确规划 | `frozen/planning/audit_joint_k5_union_exact_v0.py` | `main()` 内的深度 3、最多 5 叶精确递推 |
| STreeD-derived | `frozen/streed/rank_loss.h`、`rank_loss_cli_wide_o3.cpp` | 自定义 RankLoss 任务与实际 O3 比较入口 |

VEST-2 是上述 pinned A4 向量化深度二实现。DL8.5 是 S6 中的外部比较基线，不能代替 VEST-2。历史 **4.702×** 来自原 S6 实验，本次源码上传没有重新测量这个数值。

文件名中的 `audit_` 是原始算法文件名。仓库按用途整理为 representation、planning、retrieval、evaluation 和 streed，没有复制历史审计目录、日志或整套第三方项目。

## 服务器环境

原始服务器使用 Python 3.10.22、NumPy 1.26.4。检索环境还使用 pandas 2.3.3、SciPy 1.15.3、transformers 4.57.6 和 PyTorch 2.5.1+cu121。依赖声明见 `requirements*.txt`。原始实验中的 `fcntl`、`resource` 和 shell 入口面向 Linux。

STreeD-derived 基于官方 pystreed 1.4.0；`frozen/streed/upstream_integration.patch` 保存原始适配相对官方源码的五处注册差异。完整上游源码和模型权重不放入本仓库。适配关闭 terminal solver、similarity lower bound 和 task lower bound，具体参数保留在原始 C++ 入口中。

## 冻结输入与配置

训练 NPZ/JSON、AST44 缓存、检索矩阵、查询清单、代码嵌入和模型由服务器现有实验目录提供。S6 的冻结协议与 100 个训练案例的预期值和输入哈希保存在 `configs/`；不把这些元数据当作新测量结果。源码仓库不包含训练矩阵、受限语料、封存 Test/R12、权重或实验日志。

原始脚本保留服务器路径 `/root/autodl-tmp/work2-algorithm-innovation`。`manifests/sources.json` 映射整理后的文件和原始相对路径；别名指向完全相同的源码字节，Git 中只保存一份。

如需恢复原始目录结构，可在服务器运行：

```bash
python tools/materialize_sources.py --destination /path/to/new-work2-source-tree
```

该命令只复制源码，遇到不同内容的现有文件会停止，不接触服务器数据、实验结果或锁。服务器运行继续使用原有输入、冻结配置和原始入口；本次上传不修改已经完成的实验目录。

## SHA256 与交付文件

```bash
python tools/verify_sources.py
```

完整清单：`SHA256SUMS`。原始两个源码包的清单保存在 `manifests/*input_SHA256SUMS`。关键源码、完整目录和上传交付说明分别见 `docs/KEY_SOURCE_SHA256.md`、`docs/TREE.txt`、`docs/UPLOAD_REPORT.md`。代码授权与第三方说明见 `docs/THIRD_PARTY.md`。
