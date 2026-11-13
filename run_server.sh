#!/bin/bash
# FedCMA 完整实验(两个任务 × FedAvg/FedCMA), 冻结 BioClinicalBERT 文本分支.
# 用法: bash run_server.sh   (CPU/GPU 均可, 单 run 约 10-20 分钟)
set -e
PY=${PY:-python}

for task in ihm pheno; do
  for method in fedavg fedcma; do
    $PY main.py --task $task --method $method
  done
done

# 可选: 消融(论文 Table 2)与 FedProx
# for method in fedcma_wo_prior fedcma_wo_cl fedprox; do
#   for task in ihm pheno; do
#     $PY main.py --task $task --method $method
#   done
# done

$PY summarize.py
