"""Replace the experimental paragraph while preserving other OOXML package parts."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def paragraph_text(paragraph):
    return "".join(node.text or "" for node in paragraph.iter(W + "t"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError("Write a new revised document, preserving the original")
    rows = {r["method"]: r for r in json.loads(args.summary.read_text(encoding="utf-8"))}
    full, avg, prox = (rows[m] for m in ("fedcma", "fedavg", "fedprox"))
    metrics = lambda r: f"AUC为{r['auc_mean']:.4f}±{r['auc_std']:.4f}、AP为{r['auprc_mean']:.4f}±{r['auprc_std']:.4f}"
    text = (
        "在一个具体实施例中，进行实验验证。本实施例选取IHM-48住院死亡风险二分类预测任务，"
        "以患者入重症监护病房后前48小时的生理时序数据及对应的医疗报告文本特征作为多模态输入。"
        "训练集、验证集及测试集分别包含3072、427和1131个样本，按患者划分保证不同数据集之间无重叠。"
        "为模拟真实多中心场景，使用狄利克雷分布（α=0.5）将训练样本划分至K=5个客户端，形成非独立同分布格局，"
        "并使约50%的样本缺失一种模态。模型采用128维特征表示和4头跨模态注意力模块，优化器为Adam，"
        "学习率为0.001，批量大小为64。训练配置为T=5轮全局通信、E=3轮本地epoch，各方法均执行750次参数更新，"
        "累计处理46080个训练样本。本实施例在小样本实验中采用分层排序的课程调度：第1轮进行预热，"
        "第2、3轮在标签及缺失模式相同的样本组内，分别以0.25和0.125的排序强度融合难度次序与随机次序，"
        "第4、5轮恢复随机顺序；每个本地epoch保留全部样本并各使用一次，使各方法训练量保持一致。"
        "全部实验在随机种子42、43、44下重复执行，根据验证集平均精确率（AP）选择模型，并在测试集上计算"
        "ROC曲线下面积（AUC）及AP，以均值±标准差报告结果。"
        f"实验结果显示，本实施例方法的{metrics(full)}；FedAvg的{metrics(avg)}；FedProx的{metrics(prox)}。"
        f"与FedAvg相比，本实施例方法的AUC和AP分别提高{100*(full['auc_mean']-avg['auc_mean']):.2f}和"
        f"{100*(full['auprc_mean']-avg['auprc_mean']):.2f}个百分点；与FedProx相比，分别提高"
        f"{100*(full['auc_mean']-prox['auc_mean']):.2f}和{100*(full['auprc_mean']-prox['auprc_mean']):.2f}个百分点。"
        "上述结果表明，在所述非独立同分布及模态缺失条件下，本实施例方法取得了优于两种对比方法的平均预测性能。"
    )
    with ZipFile(args.input) as source:
        original_xml = source.read("word/document.xml")
        document = etree.fromstring(original_xml)
        paragraphs = document.find(W + "body").findall(W + "p")
        before = [etree.tostring(p) for p in paragraphs]
        matches = [i for i, p in enumerate(paragraphs) if paragraph_text(p).startswith("在一个具体实施例中，进行实验验证")]
        assert len(matches) == 1
        index = matches[0]
        paragraph = paragraphs[index]
        old = paragraph_text(paragraph)
        run = deepcopy(paragraph.find(W + "r"))
        properties = run.find(W + "rPr")
        for child in list(run):
            if child is not properties:
                run.remove(child)
        etree.SubElement(run, W + "t").text = text
        for child in list(paragraph):
            if child.tag != W + "pPr":
                paragraph.remove(child)
        paragraph.append(run)
        changes = [{"paragraph": index + 1, "before": old, "after": text}]
        # The implementation paragraph repeats the experimental training budget.
        implementation = next((i for i, p in enumerate(paragraphs) if paragraph_text(p).startswith("本领域内的技术人员应明白")))
        paragraph = paragraphs[implementation]
        old = paragraph_text(paragraph)
        replacements = {
            "每轮本地训练E=6个epoch，累计每客户端30个epoch即可达到收敛": "每轮本地训练E=3个epoch，每个客户端累计完成15个epoch的本地训练",
            "先验生成网络采用隐层64维的三层全连接结构": "先验生成网络采用128维输入、64维隐层和128维输出的两层全连接结构",
        }
        for original, replacement in replacements.items():
            targets = [node for node in paragraph.iter(W + "t") if original in (node.text or "")]
            assert len(targets) == 1, original
            targets[0].text = targets[0].text.replace(original, replacement)
        changes.append({"paragraph": implementation + 1, "before": old, "after": paragraph_text(paragraph)})
        unchanged = [i for i, p in enumerate(paragraphs) if etree.tostring(p) != before[i]]
        assert unchanged == [index, implementation], unchanged
        xml = etree.tostring(document, xml_declaration=True, encoding="UTF-8", standalone=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(args.output, "w") as target:
            for item in source.infolist():
                target.writestr(item, xml if item.filename == "word/document.xml" else source.read(item.filename))
    with ZipFile(args.input) as source, ZipFile(args.output) as target:
        assert source.namelist() == target.namelist()
        assert all(source.read(name) == target.read(name) for name in source.namelist() if name != "word/document.xml")
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps({"original_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "revised_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "changed_paragraphs": [i + 1 for i in unchanged], "other_paragraphs_unchanged": True,
        "other_package_parts_unchanged": True, "changes": changes}, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "changed_paragraphs": [i + 1 for i in unchanged], "experiment_characters": len(text)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
