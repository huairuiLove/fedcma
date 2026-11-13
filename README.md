# FedCMA / FedPT 复现

论文: *Federated Personalized Learning for Multimodal Medical Data: FedPT*
(最新版题名 FedCMA — Federated Curriculum learning with Modal-Aware information
aggregation), ICASSP 投稿, 源文件在 `../fedptEN.tex` 与
`../../论文/paper/Federated_Personalized_Learning_for_Multimodal_Medical_Data__FedPT-17.pdf`。

骨干模型: MultimodalMIMIC (Xinlu Zhang et al., ICML'23,
https://github.com/XZhang97666/MultimodalMIMIC), 时间注意力编码器
`multiTimeAttention` 原样移植于 `module.py`。

## 数据(全部来自本机磁盘 /Volumes/TiPlus7100s/mimic)

- 时序: `mimic-iv-extracted/{in-hospital-mortality,phenotyping}` — MedFuse 管线
  从 MIMIC-IV 3.1 抽取的 17 变量不规则时序 benchmark。IHM 每 48h 窗口一个样本
  (episode k = 入住后 [48(k-1), 48k) 小时, CSV 时间重置为 0), Pheno 每 24h 一个。
- 文本: `mimic-cxr-jpg_2.1.0/mimiccxr_reports` — 放射报告, 按
  `pXX/p<subject>/s<study>.txt` 组织, 用 `mimic-cxr-2.0.0-metadata.csv` 的
  StudyDateTime 对齐到每个 episode 的时间窗。
- cohort = 窗口内至少 1 份报告的 episode(双模态齐备的基础数据, 与《输入数据.docx》
  "按比例选择部分数据, 然后 50% 概率去除一个模态"一致)。本盘 MIMIC-CXR 与
  MIMIC-IV ICU 人群交集 31%, 因此覆盖率 IHM 15.3% / Pheno 9.8% — 数据现实。

| 任务 | train | val | test | 笔记/样本 |
|---|---|---|---|---|
| IHM (48h 死亡率, 二分类) | 4,467 | 427 | 1,131 | 1.93 |
| Pheno (24h 表型, 25 标签) | 5,997 | 613 | 1,526 | 1.30 |

患者级切分零重叠(已验证)。

## 管线

```
prepare_data.py    # 报告-episode 时间窗关联 -> data/{task}/cohort_{split}.csv
build_pickles.py   # p2x 张量(17ch 时序+归一化+标签+笔记) -> {split}_p2x.pkl
embed_notes.py     # 冻结 BioClinicalBERT CLS 嵌入 -> notes_{split}.npz
main.py            # 联邦实验入口
  module.py        # multiTimeAttention / TimeEmbed / TransformerEncoder (骨干移植)
  model.py         # FedCMA 模型(先验+生成器+跨模态融合, 论文 Eq.2-5)
  dataset.py       # 缺失掩码/难度分数/Dirichlet 划分/reg_ts 前向填充
  fed.py           # 联邦训练循环(Eq.6-8: FedAvg聚合+先验参考+δ插值+课程学习)
  metrics.py       # AUC/AUPRC; micro/macro AP & AUC
```

## 方法实现要点(对照论文)

- **模态先验+生成器** (Sec.3.2.1): 每客户端每模态一个 128 维先验 p_k^(m);
  缺失时特征 = G^(m)(p_k^(m)) (2 层 MLP 生成器, 全局共享); 可用模态用编码器特征。
- **损失** (Eq.5): L_task + λ_align·L_align + λ_reg·‖p‖²; L_align 只在双模态
  齐备样本上对齐 h^(m) 与 G(p^(m))(h detach)。
- **聚合** (Eq.6-8): 骨干 FedAvg; 先验按样本量加权成参考 p̄; 本地
  p_k ← (1-δ_t)·p_k + δ_t·p̄, δ_t 随轮数 0.2→1.0 线性。
- **课程学习** (Sec.3.2.3): S_i = ξ·P_i + (1-ξ)·C_i, P_i=1-(I_ts+I_txt)/2,
  C_i=客户端内归一化观测数(fedptEN.tex Eq.9-11); 每轮只用难度最低 ρ_t 部分,
  ρ_t: 0.6→1.0 线性。
- **缺失模拟**: q=0.5/模态伯努利, 双缺重采样; 掩码 seed=42 固定,
  所有方法与训练/测试共用同一套(论文 Sec.4.1)。
- 方法开关: `fedavg` `fedprox`(μ=0.01) `fedcma` `fedcma_wo_prior` `fedcma_wo_cl`
  (缺失模态基线以零特征输入)。
- 评估: (a) 各客户端用自己先验在本端测试分片评估后按样本量加权(主指标,
  个性化 FL 惯例); (b) 全局模型+参考先验在完整测试集(参考)。按 val 最优轮选点。

## 与论文的已知偏差

1. 文本编码器冻结(BioClinicalBERT CLS 预计算嵌入), 论文骨干微调
   Clinical-Longformer — 设定选择, 训练只需 CPU/轻量 GPU。
2. K=5、λ_align=0.5、λ_reg=1e-3、ρ₀=0.6、δ₀=0.2、μ=0.01 为自行设定(论文未给),
   可通过命令行调整(含 --delta_max/--align_detach 等调优项)。
3. 融合在特征向量级(论文公式即向量级), 骨干原文为 48 位置序列级跨注意力。
4. MMD 损失未实现(终版论文已删除)。
5. lr=4e-4/batch=32/layers=3/embed_dim=128 等取骨干 run.sh 默认值。

## 运行

### 本文件夹是自包含的(可整体打包搬服务器)

```
FedCMA/                 # 打包这个文件夹即可
├── *.py                # 全部代码(数据准备脚本仅在重造数据时需要)
├── configs/            # 17 通道定义与分类值映射
├── data/ihm  data/pheno   # 训练所需全部数据: cohort/p2x(含笔记原文)/嵌入/统计
├── runs/               # 本机已跑结果与日志(可删)
├── requirements.txt
└── run_server.sh       # 服务器一键脚本
```

注意: 训练/评估只需要 `data/` 下的 p2x 与嵌入文件; `prepare_data.py`
`build_pickles.py` 只在要从原始 MIMIC 重建数据时使用(路径常量在文件顶部,
原始数据未随包迁移)。

### 服务器(CPU/GPU 通用)

文本分支为**冻结的 BioClinicalBERT**(CLS 嵌入已预计算, 随 `data/` 一起分发),
训练只更新时序/融合/先验网络(~2.4M 参数), 无需下载大模型、无 GPU 也能跑:

```bash
pip install -r requirements.txt
bash run_server.sh                    # ihm/pheno × fedavg/fedcma
python main.py --task ihm --method fedcma          # 单个实验
python main.py --task ihm --method fedcma --smoke    # 2 分钟冒烟
python summarize.py                                  # 汇总结果表
```

### 运行(本机)
