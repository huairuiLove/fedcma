# 本机复现记录(Mac M4 Max, 2026-11-15)

分支 `delivery/batch3-patent-20261113`,代码 38/38 测试通过。
命令:

```bash
python run_experiments.py --out_dir runs_repro/b3main \
  --methods fedavg fedprox fedcma --seeds 42 43 44 \
  --curriculum_schedule stratified_order --warmup_rounds 1 \
  --curriculum_strength 0.25 --full_data_rounds 2 \
  --equal_training_budget --device cpu --threads 8
```

MPS 不支持该管线的确定性算子(`index_put_with_accumulate`),按交付 README
的非 CUDA 路径用 CPU;9 次训练共约 8 分钟。

## 与交付值对比(test_global, 3 种子均值±样本标准差)

| 方法 | 交付(RTX 5090, torch2.14+cu130) | 本机(M4 Max CPU, torch2.13) | 均值差 |
|---|---|---|---|
| FedAvg | AUC 0.7287±0.0078 / AP 0.3185±0.0223 | 0.7269±0.0108 / 0.3198±0.0101 | −0.002 / +0.001 |
| FedProx | 0.7196±0.0098 / 0.3190±0.0139 | 0.7149±0.0118 / 0.3285±0.0101 | −0.005 / +0.010 |
| FedCMA | 0.7463±0.0043 / 0.3479±0.0008 | 0.7416±0.0142 / 0.3499±0.0044 | −0.005 / +0.002 |

结论:全部方法差值 ≤0.01(小于种子间标准差),方法排序一致;
FedCMA 对 FedAvg 的优势(本机 +1.5pt AUC / +3.0pt AP)独立复现。

逐 run 明细见 `runs_repro/b3main/completed_runs.json`(checkpoints 与
prepared.pt 体积大,未入库,可用上述命令从 `data/ihm` 一键重建)。
