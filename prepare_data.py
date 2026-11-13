"""Step 1: link MIMIC-CXR radiology reports to MIMIC-IV ICU stay episodes.

Episodes in the extracted benchmark are 48h (IHM, always episode1) or 24h
(Pheno, episode k covers [24*(k-1), 24*k) hours after ICU intime; the episode
timeseries CSV hours are rebased to the episode start).
For each episode we collect radiology studies performed inside its window and
rebase their offsets to the episode start.

Cohort for FedCMA = episodes with >=1 in-window report (both modalities
naturally present; sample-level missingness is simulated later with q=0.5).

Outputs data/{task}/cohort_{split}.csv:
  stay, stay_id, subject_id, labels..., note_offsets_h (;;-joined, hours after
  episode start), report_paths (;;-joined)
"""
import os
import re
import sys
import pandas as pd

MIMIC_ROOT = "/Volumes/TiPlus7100s/mimic"
META_CSV = f"{MIMIC_ROOT}/mimic-cxr-jpg_2.1.0/mimic-cxr-2.0.0-metadata.csv"
REPORTS_DIR = f"{MIMIC_ROOT}/mimic-cxr-jpg_2.1.0/mimiccxr_reports"
EXTRACT = f"{MIMIC_ROOT}/mimic-iv-extracted"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TASKS = {"ihm": {"dir": "in-hospital-mortality", "window": 48.0},
         "pheno": {"dir": "phenotyping", "window": 24.0}}


def build_studies():
    meta = pd.read_csv(META_CSV, usecols=[
        "subject_id", "study_id", "StudyDate", "StudyTime"])
    meta = meta.dropna(subset=["StudyDate", "StudyTime"])
    meta = meta.sort_values("study_id").groupby("study_id", as_index=False).first()
    meta["StudyDateTime"] = pd.to_datetime(
        meta["StudyDate"].astype(str).str.split(".").str[0]
        + " " + meta["StudyTime"].astype(float).astype(int).map(lambda x: f"{x:06d}"),
        format="%Y%m%d %H%M%S", errors="coerce")
    meta = meta.dropna(subset=["StudyDateTime"])

    def report_path(row):
        sid = str(int(row.subject_id))
        return os.path.join(REPORTS_DIR, f"p{sid[:2]}", f"p{sid}",
                            f"s{int(row.study_id)}.txt")
    meta["report_path"] = meta.apply(report_path, axis=1)
    meta = meta[meta["report_path"].map(os.path.isfile)]
    print(f"studies with readable report file: {len(meta)}")
    return meta


def link(task, meta):
    cfg = TASKS[task]
    W = cfg["window"]
    stays = pd.read_csv(f"{EXTRACT}/root/all_stays.csv",
                        usecols=["subject_id", "stay_id", "intime"])
    stays["intime"] = pd.to_datetime(stays["intime"])
    meta_s = meta[["subject_id", "study_id", "StudyDateTime", "report_path"]]

    m = stays.merge(meta_s, on="subject_id", how="inner")
    m["hours_after_intime"] = (m["StudyDateTime"] - m["intime"]).dt.total_seconds() / 3600.0

    os.makedirs(f"{OUT_DIR}/{task}", exist_ok=True)
    for split in ["train", "val", "test"]:
        lf = pd.read_csv(f"{EXTRACT}/{cfg['dir']}/{split}_listfile.csv")
        eps = lf["stay"].map(lambda s: int(re.search(r"episode(\d+)_timeseries", s).group(1)))
        lf["ep_start"] = (eps - 1) * W
        j = lf.merge(m, on="stay_id", how="inner")
        j = j[(j["hours_after_intime"] >= j["ep_start"])
              & (j["hours_after_intime"] < j["ep_start"] + W)].copy()
        j["offset"] = j["hours_after_intime"] - j["ep_start"]
        j = j.sort_values(["stay", "offset", "study_id"])
        g = j.groupby("stay").agg(
            note_offsets_h=("offset", lambda s: ";".join(f"{x:.3f}" for x in s)),
            report_paths=("report_path", ";".join))
        cohort = lf.merge(g, on="stay", how="inner")
        if task == "ihm":
            print(f"[{task}/{split}] with notes: {len(cohort)}/{len(lf)} "
                  f"({100*len(cohort)/len(lf):.1f}%), pos rate {cohort['y_true'].mean():.4f}")
        else:
            print(f"[{task}/{split}] with notes: {len(cohort)}/{len(lf)} "
                  f"({100*len(cohort)/len(lf):.1f}%)")
        cohort.to_csv(f"{OUT_DIR}/{task}/cohort_{split}.csv", index=False)


if __name__ == "__main__":
    tasks = sys.argv[1:] or list(TASKS)
    meta = build_studies()
    for t in tasks:
        link(t, meta)
