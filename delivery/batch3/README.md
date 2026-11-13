# 第三批实验交付

本说明对应本地交付包的根目录。Git 分支中的 `delivery/batch3/` 保存组装与复现工具；完整数据、模型及专利文件在独立 ZIP 中提供。

专利中的实验段采用第三批主实验，比较 FedCMA、FedAvg 和 FedProx，三个随机种子为 42、43、44。正文不设置消融实验段。原始实验档案完整保留，方便后续查阅和复核。

| 方法 | AUC（均值 ± 标准差） | AP（均值 ± 标准差） |
| --- | --- | --- |
| FedAvg | 0.7287 ± 0.0078 | 0.3185 ± 0.0223 |
| FedProx | 0.7196 ± 0.0098 | 0.3190 ± 0.0139 |
| FedCMA | 0.7463 ± 0.0043 | 0.3479 ± 0.0008 |

标准差采用三个种子的样本标准差。指标来自固定测试集上的全局模型，最佳轮次由验证集 AP 选择；代码中的 `auprc` 字段实际计算的是 average precision。

## 文件位置

- `patent/`：替换实验段后的专利 Word 文件和替换段落文本。
- `main_results/`：正文使用的三种方法汇总和逐种子结果。
- `code/`：第三批开始训练时保存的完整代码快照，逐文件与原快照哈希一致。
- `input_data/ihm/`：本地已有的 IHM 处理后输入文件，包含时序、标签和报告嵌入。
- `records/experiments/batch_03_ihm_20261113/`：第三批完整原始档案。
- `verification/`：本次交付时进行的数据重建、模型重放和独立复训核验。
- `package_manifest.json`：交付文件的 SHA-256 清单；压缩包旁另有整个 ZIP 的校验值。

第三批原始档案含 18 次正式实验、12 次仅使用验证集的候选配置实验和 1 次独立复训，共 31 次运行、23,250 条训练步记录、155 份逐轮检查点及 31 份最佳模型。每次运行均保存配置、依赖版本、源代码及其哈希、逐步损失、逐轮指标、样本训练顺序、模型与先验、随机数状态和预测记录。候选实验没有测试集预测；正式实验与复训保存逐样本测试概率、标签及分组指标。旧档案中的绝对路径用于记录当时的运行位置，下面的复现入口会使用解压后的相对位置。

交付时额外进行的种子 42 复训单独存放在 `verification/retraining_s42/`，不计入上述原始 31 次实验。其全部轮次的模型、先验、验证预测、训练顺序及最终测试预测均与第三批原记录完全一致。数据重建检查也得到相同的缓存文件 SHA-256。

## 环境与复现

原运行环境为 Python 3.12.15、PyTorch 2.14.1+cu130、CUDA 13.0，显卡为 RTX 5090 D v2。依赖列表在 `code/requirements.txt`，原始完整环境在各运行目录的 `environment.json` 和 `environment-freeze.txt`。本机可直接使用已有的 `D:\zhuanli\fedcma\.venv\Scripts\python.exe`，无需重装环境。

在交付目录打开 PowerShell。下列 `python` 可替换为现有环境解释器的完整路径。

```powershell
# 校验所有原始档案、代码与数据；只做哈希检查时不需要训练依赖。
python verify_package.py

# 重放全部 31 个最佳检查点，检查预测是否与原记录一致。
python verify_package.py --replay --output replay_check.json

# 重跑正文的 3 种方法 × 3 个种子，共 9 次正式实验。
python reproduce.py --suite main --output reproduced_main --compare

# 重跑包括消融在内的 18 次正式实验。
python reproduce.py --suite formal --output reproduced_formal --compare

# 重跑第三批全部 31 次实验，包括验证集候选和独立复训。
python reproduce.py --suite all --output reproduced_all --compare

# 仅重跑本文方法的种子 42。
python reproduce.py --suite main --methods fedcma --seeds 42 --output reproduced_s42 --compare

# 从包内已有的处理后输入重新构造数据，并逐张量对照第三批缓存。
python rebuild_data.py --output rebuilt_data
```

所有重跑结果写入新的目录，原始记录保持不变。`--compare` 要求模型权重、各轮验证预测、样本顺序及测试预测与原记录完全一致，适合相同硬件和依赖环境。其他设备可去掉 `--compare`；无 CUDA 时加 `--device cpu`。CPU 或不同 CUDA 环境下可能存在数值差异。

若需要新建环境，可使用 Python 3.12 创建 venv，再安装 `code/requirements.txt`，并安装与原运行一致的 CUDA 版 PyTorch。交付包包含数据、代码和记录，不包含 Python 环境目录、CUDA 驱动及第三方安装包。

## 本批配置

IHM-48 使用入重症监护病房后前 48 小时的生理时序和医疗报告嵌入。训练、验证、测试样本数分别为 3,072、427、1,131，患者划分无交集。5 个客户端采用 Dirichlet α=0.5 的划分；约 50% 的样本缺失一种模态，即约 25% 的模态条目缺失。

统一使用 Adam、学习率 0.001、批量 64、5 轮通信、每轮 3 个本地 epoch；每次正式运行处理 46,080 个训练样本，执行 750 次参数更新。FedProx 的 μ=0.01。特征维度 128，时序编码器为轻量一维卷积，文本采用已有的 768 维嵌入，融合模块为 4 头注意力。先验生成器为 128→64→128，特征对齐与先验正则系数均为 0.001。

第三批课程按标签与缺失模式分层调整样本顺序：第 1 轮预热，第 2、3 轮排序强度分别为 0.25、0.125，第 4、5 轮恢复随机顺序。每个本地 epoch 中每条训练样本均使用一次，每个小批量的标签及缺失模式数量与对应随机序列保持一致。候选配置的选择过程保存在 `study_protocol.json`、`candidate_selection.json` 和 `validation_search/`。

`code/` 中的历史批次驱动脚本保留了原始目录引用；独立解压后应使用本目录的 `reproduce.py`。该入口调用未经修改的第三批训练代码，仅替换数据和输出路径，因此不依赖第一、第二或第四批文件。
