"""Every training run needed for the journal revision, split across the 4 of us.

    python make_patient_split.py                 # once per machine (builds ../../data_patient)
    python revision_runs.py --who ahmed --list   # show my jobs
    python revision_runs.py --who ahmed          # run them (resumable: just run again)

who = ahmed | abdelrahman | malak | jana   (each keeps the models they trained before,
so pretrained weights are already cached and micro-batches already fit their GPU)

All runs use the PATIENT-DISJOINT split (../../data_patient) and the leakage-free
'no_confirm' feature set unless stated, 15 epochs, identical protocol:
  A) all 14 models, seed 42                         -> new headline benchmark
  B) ablations: image_only / no_melanocytic         -> metadata contribution + shortcut test
  C) Tier-1 models with seeds 1 and 2               -> mean +- std, ranking stability
Outputs: unified/outputs_revision/<model>__<featureset>__<tag>/ (results.json, predictions.npz, best.pt)
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_all_local import VRAM_4GB  # noqa: E402

PATIENT = os.path.normpath(os.path.join(HERE, "..", "..", "data_patient"))
TIER1 = ["vit_early_fusion", "hybrid_vit_mlp", "gnn_maxvit", "davit_gated", "segformer_laca"]


def J(model, fs="no_confirm", seed=None, tag="patient"):
    return dict(model=model, fs=fs, seed=seed, tag=tag)


def seeds(m):
    return [J(m, seed=s, tag=f"patient_s{s}") for s in (1, 2)]


# order inside each list = priority (headline first, then ablations, then seeds)
PLAN = {
    "ahmed": [J("vit_early_fusion"), J("gnn_maxvit"), J("swinv2_gat"), J("unet_crossattn"),
              J("gnn_maxvit", "no_melanocytic")] + seeds("gnn_maxvit"),
    "abdelrahman": [J("hybrid_vit_mlp"), J("convnext_bicross"), J("effnetv2_crossattn"),
                    J("dinov2_crossattn"), J("biovil_lcsa"),
                    J("gnn_maxvit", "image_only")] + seeds("hybrid_vit_mlp"),
    "malak": [J("davit_gated"), J("maxvit_crossattn"), J("convnext_bilinear"), J("swinv2_tabtransformer"),
              J("vit_early_fusion", "image_only"), J("vit_early_fusion", "no_melanocytic"),
              J("davit_gated", "image_only"), J("davit_gated", "no_melanocytic")] + seeds("davit_gated"),
    "jana": [J("segformer_laca")] + seeds("segformer_laca") + seeds("vit_early_fusion"),
}
# rough single-run minutes measured in the first sweep (for the --list estimate)
MIN = {"vit_early_fusion": 49, "hybrid_vit_mlp": 83, "gnn_maxvit": 61, "davit_gated": 21, "segformer_laca": 114,
       "maxvit_crossattn": 40, "convnext_bicross": 26, "swinv2_gat": 68, "effnetv2_crossattn": 26,
       "convnext_bilinear": 22, "swinv2_tabtransformer": 52, "biovil_lcsa": 27, "dinov2_crossattn": 28,
       "unet_crossattn": 15}


def name_of(j):
    return f"{j['model']}__{j['fs']}__{j['tag']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", required=True, choices=list(PLAN))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out-root", default=os.path.join(HERE, "outputs_revision"))
    a = ap.parse_args()
    assert os.path.exists(os.path.join(PATIENT, "CUSTOM_SPLIT")), \
        "run  python make_patient_split.py  first (builds ../../data_patient)"
    os.makedirs(a.out_root, exist_ok=True)
    jobs = PLAN[a.who]
    left = [j for j in jobs if not os.path.exists(os.path.join(a.out_root, name_of(j), "results.json"))]
    print(f"{a.who}: {len(jobs)} jobs, {len(left)} left, ~{sum(MIN[j['model']] + 5 for j in left) / 60:.1f} h on the "
          f"first-sweep GPU speed", flush=True)

    for i, j in enumerate(jobs, 1):
        nm = name_of(j)
        done = os.path.join(a.out_root, nm, "results.json")
        print(f"[{i}/{len(jobs)}] {nm} {'(done)' if os.path.exists(done) else ''}", flush=True)
        if a.list or os.path.exists(done):
            continue
        micro, t0 = VRAM_4GB.get(j["model"], 16), time.time()
        log_p = os.path.join(a.out_root, f"log_{nm}.txt")
        for attempt in range(4):
            cmd = [sys.executable, "-u", os.path.join(HERE, "run.py"), "--model", j["model"],
                   "--feature-set", j["fs"], "--data-root", PATIENT, "--out-root", a.out_root,
                   "--epochs", str(a.epochs), "--micro-batch", str(micro), "--workers", "0",
                   "--no-xai", "--tag", j["tag"]] + (["--seed", str(j["seed"])] if j["seed"] is not None else [])
            with open(log_p, "a", encoding="utf-8") as log:
                log.write(f"\n=== attempt {attempt + 1} micro={micro}\n"); log.flush()
                rc = subprocess.call(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            if rc == 0 and os.path.exists(done):
                break
            tail = open(log_p, encoding="utf-8", errors="replace").read()[-3000:].lower()
            if "out of memory" in tail and micro > 1:
                micro //= 2; print(f"    OOM -> micro={micro}", flush=True); continue
            print(f"    rc={rc} -> retry/resume ({attempt + 2}/4)", flush=True)
        if os.path.exists(done):
            t = json.load(open(done, encoding="utf-8"))["test"]
            print(f"    ok in {(time.time() - t0) / 60:.0f} min | AUC {t['auc']:.4f}", flush=True)
        else:
            print(f"    FAILED -> send me {log_p}", flush=True)
    print("ALL DONE for", a.who, "-> zip the folder outputs_revision and send it", flush=True)


if __name__ == "__main__":
    main()
