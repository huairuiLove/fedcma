# 实验批次索引

第一至第四批使用同一份固定数据，分别保留结果。三个种子均为42、43、44；这是训练随机性比较，未改变患者划分、客户端分配或缺失掩码。

## 第四批

本次目录：`experiments/batch_04_ihm_20261113/`，配置 `configs/batch_04.json`，入口 `run_batch4.py`。课程由当前训练小批次的难度决定软权重，保留原随机顺序、每样本覆盖、批次组成及组别总权重。

状态：18次验证候选、18次正式训练及1次完整重训全部完成，训练与核验约291秒。选中预热1轮、最大权重强度0.25；完整模型AP为0.3361±0.0232，无课程为0.3509±0.0043，本批未实现稳定提升。完整结果见 `batch4_report.md`，训练与预测的进一步诊断见 `postrun_interpretation.md`。

六组验证候选×三个种子单独存于 `validation_search/`；只按验证结果选定方案后，再运行六组正式方法×三个种子。总轮数5、每轮本地epoch数3、每次46,080次样本处理和750次更新均保持一致。

`preparation/prior_batches_manifest.json` 索引前三批4,478个文件；`preparation/project_before_batch4/` 保存本次修改前的代码；`preparation/analysis/` 保存第三批实际调序幅度及验证/测试差异分析。每次训练的 `selection_round_*.npz` 新增逐样本当批BCE和课程权重，全部留存并复核。

前三批1,098,141,678字节的文件全部重新通过哈希检查。第四批保留27,750条逐步记录、185个每轮检查点和37个最佳模型（包括候选与额外重训）；正式部分仍为13,500条记录、90个每轮检查点和18个最佳模型。38项测试、独立CPU权重复算、所有模型重放和完整重训核验通过。四批逐种子AP/AUROC对照为 `four_batch_comparison.csv`。

## 第三批

本次目录：`experiments/batch_03_ihm_20261113/`，配置 `configs/batch_03.json`，入口 `run_batch3.py`。第1轮预热；课程保留全部样本，并在同标签、同缺失模式内部调整学习顺序，末期恢复随机顺序。每个epoch每个样本恰好一次，每批的标签/缺失组成保持一致。

状态：12次验证候选、18次正式训练和1次额外完整重训全部完成。初始强度0.25，最后2轮恢复普通随机顺序；正式AP为0.3479±0.0008，未稳定超过无课程组。完整结果见本目录的 `batch3_report.md`，三批逐种子对照见 `three_batch_comparison.csv`。

四组候选在 `validation_search/` 内分别保留三个种子的全部验证训练；选择准则预先记录于 `study_protocol.json`，选中后正式配置写入 `protocol.json`。正式对照共六组×三个种子；每次46,080次样本处理、750次更新。候选试验与额外重训核验不并入正式均值。

`preparation/prior_batches_manifest.json` 索引前两批全部2,768个文件；`preparation/project_before_batch3/` 保存本次修改前的代码与说明；`preparation/analysis/` 保存第二批的正例曝光和预测排名诊断。第三批结束后前两批763,246,451字节的文件全部重新通过哈希核验。

本批共保存23,250条逐步训练记录、155个每轮检查点和31个最佳模型（含验证候选及额外重训）。正式18次训练的13,500条记录、90个每轮检查点和18个最佳模型单独汇总。所有最佳模型重放预测差为0，选中候选和额外重训也逐轮精确复现；32项实现测试通过。

## 第一批

第一批原文件继续存放在原位置，避免破坏已记录的绝对路径；`experiments/batch_01/manifest.json` 将它们统一标记为“第一批”，逐个记录文件路径、字节数和SHA-256。

| 内容 | 位置 |
|---|---|
| 主实验三组 × 三种子 | `experiments/final_ihm_20261113/` |
| 四组消融 × 三种子 | `experiments/ablation_ihm_20261113/` |
| 最初验证试跑 | `experiments/dev_20261113_a/` |
| 五组验证参数搜索 | `experiments/validation_search_20261113/` |
| 完整模型重训核验 | `experiments/reproducibility_fedcma_s42/` |
| 消融启动失败记录 | `experiments/ablation_ihm_20261113_setup_failed/` |
| 修改前完整项目代码与说明 | `experiments/batch_01/project_snapshot_before_batch2/` |
| 报告 | `EXPERIMENT_REPORT.md`、`ABLATION_REPORT.md` |

对应控制台日志也全部列入第一批索引。第一批配置为`configs/local_ihm.json`：无预热，阈值0.6→1，筛选后每个入选样本训练3个epoch，因此课程组训练处理量略少。原模式仍可通过`--curriculum_schedule legacy`运行。

## 第二批

本次输出目录：`experiments/batch_02_ihm_20261113/`。再次运行`run_batch2.py`默认使用新的时间戳目录。

状态：18次正式训练和1次完整重训核验已全部完成。26项测试通过，18个最佳模型重放的最大预测差为0。正式训练每次46,080次样本处理、750次更新；共保存13,500条逐步记录、90个每轮检查点和18个最佳模型。额外重训的750条记录、5个每轮检查点及1个最佳模型另行保留。

配置为`configs/batch_02.json`，仅覆盖课程调度、预热轮数、起始阈值与训练量控制，其余设置继承第一批。固定运行六组 × 三种子：完整FedCMA、FedAvg、FedProx、去掉先验、去掉课程学习、两者都去掉。第1轮全样本预热，第2—5轮阈值0.55→0.65；每客户端每epoch的样本处理次数与批大小序列保持一致。

第二批单独保存`protocol.json`、配置、源码快照、全部日志、每轮模型与预测、实际训练索引、汇总CSV和图表。额外重训核验放在`retraining_check/`，不混入三个种子的正式汇总。完整报告为该目录下的`batch2_report.md`；两批全部方法的逐种子变化见`batch1_vs_batch2.csv`。

数据缓存SHA-256：`81e50c9e06a96e0e666bcac7af6036084aaa8fbf855277c8a1156c71ed50d80a`。

第一批结果不重命名、不覆盖。第二批训练开始前和结束后都读取第一批索引并校验全部文件。第二批不重新搜索参数，不依据其测试结果改动本批方案。

校验结果：第一批1,751个文件、542,260,187字节全部通过哈希核验；第二批未改动的四组对照逐轮精确重现第一批。详见第二批目录下的 `batch1_preservation_after.json`、`control_verification.json` 和 `retraining_verification.json`。
