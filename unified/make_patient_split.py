"""Build a PATIENT-GROUPED 70/15/15 split (no patient in more than one split).

    python make_patient_split.py                 # ../../data -> ../../data_patient

The assignment is frozen in unified/splits/patient_{train,val,test}.csv (committed),
so every teammate gets the byte-identical split regardless of sklearn version.
If those files are missing it is (re)computed with StratifiedGroupKFold(20, seed 42):
folds 0-13 train, 14-16 val, 17-19 test. Images are hard-linked (no extra disk).
A CUSTOM_SPLIT marker lets data.load_split accept the new partition sizes.
"""
import argparse, os, shutil
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(HERE, "..", "..", "data"))
ap.add_argument("--dst", default=os.path.join(HERE, "..", "..", "data_patient"))
ap.add_argument("--recompute", action="store_true")
a = ap.parse_args()
SPL = os.path.join(HERE, "splits"); os.makedirs(SPL, exist_ok=True)
frozen = {s: os.path.join(SPL, f"patient_{s}.csv") for s in ("train", "val", "test")}

frames = []
for s in ("train", "val", "test"):
    d = pd.read_csv(os.path.join(a.src, s, f"{s}.csv")); d["_src"] = s; frames.append(d)
df = pd.concat(frames, ignore_index=True)
# 'Unknown' is a placeholder (8,668 ISIC-2019 images), NOT one patient: each such image is its own group
unk = df.patient_id.isna() | (df.patient_id.astype(str) == "Unknown")
grp = df.patient_id.astype(str).where(~unk, "img_" + df.isic_id.astype(str))
strat = df["class"].astype(int) * 2 + unk.astype(int)   # balance class AND unknown-patient share

if all(os.path.exists(p) for p in frozen.values()) and not a.recompute:
    lab = {}
    for s, p in frozen.items():
        for iid in pd.read_csv(p).isic_id: lab[iid] = s
    assign = df.isic_id.map(lab).values
    assert pd.notna(assign).all(), "frozen split does not cover every image"
else:
    from sklearn.model_selection import StratifiedGroupKFold
    fold = np.empty(len(df), int)
    for f, (_, idx) in enumerate(StratifiedGroupKFold(20, shuffle=True, random_state=42).split(df, strat, grp)):
        fold[idx] = f
    assign = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))
    for s in ("train", "val", "test"):
        df.loc[assign == s, ["isic_id", "patient_id", "class"]].to_csv(frozen[s], index=False)

for s in ("train", "val", "test"):
    out = os.path.join(a.dst, s, "images"); os.makedirs(out, exist_ok=True)
    part = df[assign == s]
    keep = set(part.image_fixed)
    for old in os.listdir(out):          # drop links left by an earlier assignment
        if old not in keep: os.remove(os.path.join(out, old))
    for src, fn in zip(part._src, part.image_fixed):
        dst = os.path.join(out, fn)
        if not os.path.exists(dst):
            try: os.link(os.path.join(a.src, src, "images", fn), dst)
            except OSError: shutil.copy2(os.path.join(a.src, src, "images", fn), dst)
    part.drop(columns="_src").to_csv(os.path.join(a.dst, s, f"{s}.csv"), index=False)
    print(f"{s}: {len(part)} images ({len(part)/len(df):.1%}), {grp[assign == s][~unk[assign == s]].nunique()} known patients, "
          f"unknown-pid {unk[assign == s].mean():.3f}, melanoma rate {part['class'].mean():.3f}")
open(os.path.join(a.dst, "CUSTOM_SPLIT"), "w").write("patient-grouped StratifiedGroupKFold(20), seed 42\n")
g = {s: set(grp[(assign == s) & ~unk]) for s in ("train", "val", "test")}
ov = len(g["train"] & g["test"]) + len(g["train"] & g["val"]) + len(g["val"] & g["test"])
print("patient overlap (train/val/test pairs):", ov)
assert ov == 0
