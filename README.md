# FedCMA 本地专利实验

本项目运行IHM小规模二分类实验，比较FedCMA、FedAvg和FedProx，保存可重放的模型、训练记录与预测。旧的`runs/`保留为历史材料；新汇总程序只读取明确指定的实验目录，不使用旧稿或旧日志中的数值。

## 第三批专利交付

本次稿件采用第三批主实验结果，比较 FedCMA、FedAvg、FedProx，随机种子为 42、43、44。完整训练记录和处理后输入在本地独立交付包中保存；第一、第二和第四批继续保留在原位置。

交付说明与独立复现入口见 [delivery/batch3/README.md](delivery/batch3/README.md)。包内 `code/` 是第三批实际训练时保存的源码快照；当前仓库同时保留各批实验入口。`delivery/batch3/build_package.py` 从原始文件清单组装交付包，核对 SHA-256，并逐文件校验 ZIP 内容。解压后可直接重跑正文的 9 次实验、全部 18 次正式实验或包含验证候选在内的 31 次运行，无需第一、第二批目录。

正文报告的 FedCMA 测试 AUC 为 0.7463±0.0043、AP 为 0.3479±0.0008；FedAvg 分别为 0.7287±0.0078、0.3185±0.0223；FedProx 分别为 0.7196±0.0098、0.3190±0.0139。均为三个种子的均值和样本标准差。稿件正文只设置这三种方法的主实验比较。

技术对照见 [IMPLEMENTATION_REVIEW.md](IMPLEMENTATION_REVIEW.md)。第一批主实验结果见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

第一批补充消融见 [ABLATION_REPORT.md](ABLATION_REPORT.md)。第一批所有原始结果已保留；[BATCHES.md](BATCHES.md) 区分四批实验、配置和存储位置。原入口保持第一批设置，后续批次使用各自的专用入口。

## 第四批：保持随机顺序的在线课程权重

```powershell
.\.venv\Scripts\python.exe run_batch4.py
```

读取 `configs/batch_04.json`，先运行六组预设候选，每组三个种子，只计算验证指标。候选选择同时考虑最佳验证AP和最后两轮平均验证AP的最差配对增益。选定后冻结参数，再跑六组×三个种子的正式比较。

本批与无课程组使用完全相同的随机样本顺序和小批次，根据每个训练小批次中同标签×同缺失模式内的当前BCE名次赋予有界权重，预热后逐步提高难例权重强度。每组平均权重保持为1，保留所有样本。每次仍为46,080次样本处理、750次更新，保存逐样本BCE与实际权重；这些权重还会用独立CPU实现复算核验。前三批原文件全部保留并做哈希检查。

第三批的调序诊断见 [第四批调整前分析](experiments/batch_04_ihm_20261113/preparation/analysis/diagnosis.md)。本批课程软加权与前面实现的技术差别见 [IMPLEMENTATION_REVIEW.md](IMPLEMENTATION_REVIEW.md)。

第四批已完成18次验证候选、18次正式训练和1次完整重训。选中预热1轮、最大权重强度0.25的设置；完整模型AP为0.3361±0.0232，无课程为0.3509±0.0043，仅1/3个种子提高，seed 44明显下降。本批未实现稳定AP提升。完整表格见 [第四批报告](experiments/batch_04_ihm_20261113/batch4_report.md)，保存记录的进一步解释见 [结果诊断](experiments/batch_04_ihm_20261113/postrun_interpretation.md)。

38项测试通过，全部候选与正式模型重放及额外重训通过；实际样本顺序、每批权重和及未改动对照的逐轮等价性均核验通过。前三批4,478个文件哈希未变。第四批全部27,750条逐步记录、185个每轮检查点和37个最佳模型保留，验证候选与正式结果分别汇总。

## 第三批：保留全部样本的分层课程

```powershell
.\.venv\Scripts\python.exe run_batch3.py
```

第三批读取 `configs/batch_03.json`，先进行四组仅验证集的课程设置比较，每组种子42、43、44；按最差种子的配对验证AP增益、再按平均增益选择方案。随后运行六组正式对照，各三个种子。完整过程使用独立的 `experiments/batch_03_*/` 目录，所有候选也保留。

本批课程保留每个样本每个epoch恰好一次，只改变同标签、同缺失模式样本内部的顺序。每个小批次的标签与缺失模式计数与无课程组完全相同，保留预热和末期普通随机训练。参见 [第三批调整前诊断](experiments/batch_03_ihm_20261113/preparation/analysis/diagnosis.md)；本批采用分层排序课程，实现变化在 [技术对照](IMPLEMENTATION_REVIEW.md) 中说明。

第三批已完成12次验证候选、18次正式训练和1次完整重训，详见 [第三批完整报告](experiments/batch_03_ihm_20261113/batch3_report.md)。选中的初始排序强度为0.25，最后2轮恢复普通随机训练。完整模型AP为0.3479±0.0008，较第二批0.3453±0.0197波动明显减小，但仍低于无课程组0.3509±0.0043，仅1/3个种子超过无课程组。AUROC为0.7463±0.0043，较第二批0.7533回落，仍高于无课程组均值0.7421。本批解决了训练覆盖偏差，尚未证明课程带来稳定AP增益。

每个种子的正例训练次数均恢复为7,440，缺失正例为3,750；每批的组别计数也与无课程组一致。32项测试通过；所有正式模型和候选模型重放、额外重训及未改动对照的逐轮复现检查通过。前两批2,768个文件哈希保持不变。

## 第二批：预热与相同训练量

```powershell
.\.venv\Scripts\python.exe run_batch2.py
```

读取 `configs/batch_02.json`，复制第一批已经固定的数据与训练配置，在新的 `experiments/batch_02_时间戳/` 下运行完整模型、FedAvg、FedProx、去先验、去课程、两者都去掉六组，各三个种子。默认拒绝覆盖已有目录。

总训练仍为5轮，每轮3个本地epoch。第1轮全样本预热，第2—5轮将难度阈值从0.55线性升至0.65。上界只由训练缺失掩码和ξ计算。每个客户端每个epoch训练原始n_k次样本；课程组按随机排列循环重复入选池，直到达到相同训练量。所有组的批大小序列、更新次数和样本处理次数一致，课程评分的额外前向开销单独记录。

`selection_round_*.npz` 除完整池、分数、入选池外，第二批还保存每个客户端、每个epoch按训练顺序排列的实际样本索引。程序核验全部结果、预热轮等价性、未改动对照与第一批的逐轮一致性，并额外完整重训一次FedCMA seed 42。两批结果分别汇总，不合并计为更多随机种子。

本次第二批已完成18次正式训练及1次完整重训核验，详见 [第二批完整报告](experiments/batch_02_ihm_20261113/batch2_report.md)。完整模型平均AUROC从第一批0.7427提高到0.7533，AP从0.3400提高到0.3453。相比无课程组，三个种子的AUROC都更高；AP仅两个种子更高，平均AP仍低于无课程的0.3509。因此本次有平均指标改善，尚未得到课程学习稳定提高AP的结果。全部记录与两批配对差异均已留档。

## 本机环境

- Windows，RTX 5090 D v2，24 GB显存，约32 GB系统内存。
- 项目Python：`.venv\Scripts\python.exe`，Python 3.12，PyTorch 2.14.1加CUDA 13.0。
- 当前venv复用`D:\environment\envs\vnvc`中已有的CUDA PyTorch及基础包，项目缺少的包装在本地venv内。没有修改该已有环境。
- `requirements.txt`固定训练依赖版本；每次实验另外保存完整`environment-freeze.txt`及硬件信息。干净环境可用`setup_local.ps1 -CleanInstall -BasePython <Python3.12路径>`重新安装。

在PowerShell中运行：

```powershell
cd D:\zhuanli\fedcma
.\.venv\Scripts\python.exe -m pytest tests -q
.\run_local.ps1
```

`run_local.ps1`读取`configs/local_ihm.json`，自动创建带时间的新目录，运行三种方法和三个种子，不覆盖已有结果。若虚拟环境不存在，先运行`setup_local.ps1`。新环境安装需要联网；使用现有处理数据训练不下载模型或原始数据。

## 第一批正式配置

| 项目 | 设置 |
|---|---|
| 任务 | IHM，输入48小时窗口，预测院内死亡 |
| 样本 | 训练3,072，验证427，测试1,131 |
| 训练抽样 | 按标签分层，subset_seed=2026 |
| 模态 | 17通道时序与放射报告文本嵌入 |
| 缺失模拟 | 约50%的样本缺失一个模态，约25%的模态条目缺失 |
| 模拟客户端 | 5个，训练患者按标签Dirichlet分配，α=0.5 |
| 训练种子 | 42、43、44；数据划分与缺失掩码固定 |
| 全局轮数与本地epoch | 5轮，每轮3个本地epoch |
| 优化器与学习率 | 三方法均为Adam，lr=0.001，batch=64 |
| FedProx | μ=0.01 |
| FedCMA | 128维先验，128→64→128生成器，λ_align=0.001，λ_reg=0.001 |
| 课程与混合 | ξ=0.7，ρ从0.6升到1，δ从0.2升到0.6 |
| 选模 | 验证AP最高的轮次，平分保留较早轮次 |
| 主要评价 | 全局模型在全部测试样本上的AUROC、AP |
| 次要评价 | 个性化先验的合并预测；完整和缺失样本分组 |

两种基线采用相同编码器与融合结构、缺失特征零填充，均不加入FedCMA机制。这里的FedAvg/FedProx使用本地Adam更新，具体设置已随结果留存。FedCMA做了5组专用参数的验证集搜索，基线采用共同固定参数；详细搜索结果全部保留。这是当前配置下的比较。

## 模型与数据处理

时序分支由两层64通道一维卷积、时间池化、每通道统计特征及128维投影构成。文本分支对已有BioClinicalBERT嵌入取最后最多5份报告的均值，再训练小型投影网络。融合模块对两个模态token执行四头自注意力和注意力池化。FedCMA增加模态先验与生成器；可训练模型约34万参数，基线约31万参数，省去大型文本模型微调。

每通道异常值截断、时序标准化和文本标准化只在选定训练子集上拟合。每轮动态计算当前BCE难度，按专利的绝对阈值筛选。服务器按完整客户端训练样本量聚合，随后更新全局参考先验与个性化先验。

数据是仓库已经附带的处理结果，不是原论文代码与数据的完整重建。`prepare_data.py`、`build_pickles.py`、`embed_notes.py`保留原始准备脚本，其旧机器路径只有在提供原始数据并修改路径后才能使用；训练入口不调用它们。`module.py`保留原始时间注意力模块作参考，当前轻量模型不导入它。

## 实验目录

```text
experiments/final_ihm_20261113/
  protocol.json                   # 训练开始前确定的方案
  data/manifest.json              # 数据来源哈希、抽样、患者和客户端、归一化参数
  data/prepared.pt                # 三方法共用的数值张量与缺失掩码
  fedcma_s42/                     # 每方法每种子独立目录
    config.json
    environment.json
    environment-freeze.txt
    source/                      # 本次运行实际使用的Python源码快照
    source_sha256.json
    train.log
    steps.jsonl                  # 逐批task/align/reg/prox/total损失与梯度范数
    history.json                 # 每轮训练均值和验证指标
    selection_round_01.npz       # 逐轮评分与实际入选样本
    checkpoints/round_01.pt      # 每轮全局权重、本地先验及参考先验
    best.pt
    val_round_01.csv
    test_predictions.csv
    test_personalized_predictions.csv
    result.json
  per_seed.csv                   # 全部种子，包含选中的轮次
  summary.csv                    # 均值与样本标准差
  summary.json
  training_curves.png
  test_comparison.png
  verification.json              # 逐次重放和指标复算核验
```

参数搜索位于`experiments/validation_search_20261113/`，最初验证试跑位于`experiments/dev_20261113_a/`。这些目录同样保留模型、配置和所有训练日志。搜索与试跑设置了`validation_only`，没有输出测试集预测。

每轮重新创建本地Adam优化器，因此检查点是全局轮次结束状态，不包含批次中途恢复所需的优化器动量。当前支持完整训练重跑和检查点预测重放，不提供中断批次续训命令。

## 复核与单次运行

```powershell
# 复核全部正式结果，重新加载所有最佳模型，重算预测和指标
.\.venv\Scripts\python.exe verify_results.py experiments\final_ihm_20261113

# 只重放一个检查点
.\.venv\Scripts\python.exe evaluate_checkpoint.py experiments\final_ihm_20261113\fedcma_s42

# 用同一份缓存数据重新训练一个方法，输出路径必须是新目录
.\.venv\Scripts\python.exe main.py --prepared experiments\final_ihm_20261113\data\prepared.pt --out_dir experiments\replay_fedcma_s42 --method fedcma --seed 42 --rho0 0.6

# 重新生成表格与曲线
.\.venv\Scripts\python.exe summarize.py experiments\final_ihm_20261113
```

可用`--device cpu`运行单次训练。固定版本、设备和设置下使用确定性算法；跨设备或跨PyTorch版本的逐位一致性不作保证。运行目录中包含明确的输入哈希和实际软件环境，便于定位差异。

## 补充消融

固定主实验的数据与参数，运行完整模型、去掉先验模块、去掉课程学习、两者都去掉四组，各三个种子。先验模块包括先验向量、生成器、对齐损失、先验正则与共享；去掉课程时每轮使用全体本地训练样本。程序同时核对完整模型重现原主实验、两者都去掉重现FedAvg。

```powershell
.\.venv\Scripts\python.exe run_ablations.py
.\.venv\Scripts\python.exe verify_results.py experiments\ablation_ihm_20261113
.\.venv\Scripts\python.exe report_ablation.py experiments\ablation_ihm_20261113
```

第一次命令默认创建时间戳目录。已完成结果位于`experiments/ablation_ihm_20261113/`，保留全部12次训练及6组等价性比较。本次没有重新搜索消融参数。当前结果中先验模块有正向贡献，而去掉课程的AP更高，详见完整报告。

## 文件职责

| 文件 | 职责 |
|---|---|
| `dataset.py` | 现成数据读取、修复填充、训练集拟合、划分与缓存 |
| `model.py` | 轻量骨干、双模态注意力、先验生成与联合损失 |
| `fed.py` | 三种方法的训练、课程调度、聚合、选模和结果保存 |
| `main.py` | 单次训练参数入口 |
| `run_experiments.py` | 三种方法及全部指定种子的组合运行 |
| `methods.py` | 先验、课程和近端项的独立开关 |
| `run_ablations.py` | 复用已固定主实验配置，运行四组消融并复核 |
| `curriculum.py` | 旧调度、新预热调度以及固定训练量采样 |
| `run_batch2.py` | 第二批六组实验、第一批留档核验、逐轮对照和重训检查 |
| `report_batch2.py` | 第二批完整结果、两批配对差异与课程采样诊断 |
| `analyze_curriculum_ap.py` | 从已保存索引与预测核对正例训练次数和排名差异 |
| `run_batch3.py` | 第三批验证候选选择、六组正式比较、前两批保留与等价性核验 |
| `report_batch3.py` | 第三批完整结果、三批逐种子差异和覆盖核验 |
| `analyze_batch3_curriculum.py` | 读取第三批真实调序幅度与验证/测试差异 |
| `run_batch4.py` | 第四批在线加权课程的验证选择、六组训练及复现核验 |
| `report_batch4.py` | 第四批全部结果、四批对照、权重与训练量汇总 |
| `report_ablation.py` | 由真实结果生成完整消融表与可选补充文字 |
| `tune_validation.py` | 留档的五组FedCMA验证集参数搜索 |
| `metrics.py` | 合并AUROC、AP和BCE计算 |
| `summarize.py` | 生成表格与可导出的静态曲线 |
| `evaluate_checkpoint.py` | 最佳模型预测重放 |
| `verify_results.py` | 整套结果一致性检查 |
| `tests/test_training_invariants.py` | 缺失隔离、公式、划分和注意力等关键测试 |

`experiments/`和`.venv/`已加入Git忽略规则，保留在本地。协作时可提交代码和说明；需要搬运实验时复制完整结果目录并保留原始输入文件。此次没有向远程仓库推送。
