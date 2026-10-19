"""Finite detached read-only study: CPU bootstrap and GPU inference observers."""
import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

for _variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_variable]="1"

from polypbench.io import read_json,sha256,write_json


def prepare(output):
    root=Path.cwd().resolve(); out=Path(output).resolve(); out.mkdir(parents=True,exist_ok=False)
    old=root/"runs/mssl-attribution-20261018-r2"; campaign=read_json(old/"protocol.json")
    if read_json(old/"state.json")["status"]!="completed": raise RuntimeError("Four-group experiments incomplete")
    paths={label:str((root/"runs/polypgen-zeroshot-20261018"/label/"pooled/predictions.bbox.json")
                     if label in ("baseline","mssl") else (old/"external"/label/"pooled/predictions.bbox.json"))
           for label in campaign["runs"]}
    data=root/"data/polypgen_zeroshot_stills_v2"; old_external=read_json(root/"runs/polypgen-zeroshot-20261018/protocol.json")
    source=out/"source"; source.mkdir()
    shutil.copytree(root/"polypbench",source/"polypbench",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    (source/"tests").mkdir(); shutil.copy2(root/"tests/test_mssl_analysis.py",source/"tests/test_mssl_analysis.py")
    # Inspect public metadata fields; do not infer subject IDs from filename tokens.
    metadata=Path(old_external["image_root"])/"dataDetails_PolypGen_SingleFrames"
    audit={"csv":[],"mapping_status":"unavailable","patient_id_inferred_from_filename":False}
    for path in sorted(metadata.glob("*.csv")):
        with path.open(encoding="utf-8-sig") as stream:
            reader=csv.DictReader(stream); count=sum(1 for _ in reader)
            audit["csv"].append({"file":str(path),"sha256":sha256(path),"fields":reader.fieldnames,"rows":count})
        if any(any(token in field.lower() for token in ("patient","subject","video","sequence")) for field in reader.fieldnames):
            raise RuntimeError("Potential grouping metadata found; inspect it before choosing image-level bootstrap")
    audit["conclusion"]="已审计附带单帧 CSV：没有可靠患者/视频分组字段，C6 未提供同类 CSV；不从文件名猜测患者。使用条件于现有六中心的图像级分层配对 Bootstrap，并明确患者内相关未知。"
    write_json(out/"group_audit.json",audit)
    inputs={"annotations":str(data/"provided_boxes.coco.json"),"sensitivity_annotations":str(data/"without_internal_duplicates.coco.json"),
            "image_manifest":str(data/"image_manifest.json"),"predictions":paths}
    flat=[inputs[k] for k in ("annotations","sensitivity_annotations","image_manifest")]+list(paths.values())
    training={}
    from polypbench.zeroshot import validate_frozen_training
    for label,run_id in campaign["runs"].items():
        trained=root/"runs"/run_id
        cfg,checkpoint,digest=validate_frozen_training(trained)
        training[label]={"experiment":str(trained/"experiment.json"),"checkpoint":str(checkpoint),"checkpoint_sha256":digest,
                         "experiment_sha256":sha256(trained/"experiment.json")}
    contract={"version":"MSSL V1, frozen actual trained definition", "mask_support":"raw thresholded GT semantic masks, original single-box eligibility",
        "loss":"mean eligible GT / mean four sides of 0.5*(G(d)-G(0))^2",
        "d":"(predicted side-provided GT side)/max(provided GT axis size,4px)",
        "pi":"normalize axial GT foreground mass * exp(-abs(relative support position)/0.15)",
        "G":"sum pi*sigmoid((d-relative support position)/0.05)",
        "lambda":1.,"tau":.05,"bandwidth":.15,"min_axis_pixels":4.,
        "application":"final native decoder matched boxes only; original assignment, L1/GIoU and mask losses retained",
        "inference":"original MaskDINO graph, predicted masks and class*mask-quality scoring retained; no GT masks",
        "trained_source_root":read_json(root/"runs"/campaign["runs"]["mssl"]/"experiment.json")["project_root"],
        "new_training":False,"new_parameters":False,"target_tuning":False}
    write_json(out/"mssl_v1_contract.json",contract)
    protocol={"created_at":dt.datetime.now(dt.timezone.utc).isoformat(),"workspace":str(root),"source_root":str(source),
        "inputs":inputs,"input_sha256":{p:sha256(p) for p in flat},"training":training,"image_root":old_external["image_root"],
        "group_audit_sha256":sha256(out/"group_audit.json"),"method_contract_sha256":sha256(out/"mssl_v1_contract.json"),
        "source_sha256":{p.relative_to(source).as_posix():sha256(p) for p in source.rglob("*") if p.is_file()},
        "rules":{"new_training":False,"loss_changes":False,"hyperparameter_search":False,"bootstrap_replicates":2000,
            "bootstrap_seed":20261019,"cpu_workers":4,"bootstrap_unit":"image within center; paired across all models",
            "copy_ids":"independent per-copy image and GT IDs, canonical original-image order",
            "thresholds":[.5,.75,.8,.85,.9,.95],"top_k":[1,5,10,100],"all_query_geometry":300,
            "negative_score_threshold":.5,"coverage_matching":"maximum cardinality, threshold constrained",
            "cohorts":"fixed1473; leave-one-center6; exclude-four-conflicting-duplicates sensitivity",
            "counterfactual":"same original100 boxes/candidates, class-only scores for diagnostic only; deployment unchanged"}}
    write_json(out/"protocol.json",protocol)
    env=os.environ.copy(); env["PYTHONPATH"]=str(source)+os.pathsep+env.get("PYTHONPATH",""); env["PYTHONUNBUFFERED"]="1"
    with (out/"pipeline.log").open("w",encoding="utf-8") as log:
        child=subprocess.Popen([sys.executable,"-m","polypbench.analysis.study","run","--output",str(out)],
            cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/"launcher.json",{"pid":child.pid,"detached":True,"output":str(out)})
    print(f"Detached analysis PID {child.pid}: {out}",flush=True)


def cpu(output):
    from .coco_cache import CocoCache,metrics
    from .behavior import run as behavior
    from .uncertainty import run as bootstrap
    import numpy as np
    out=Path(output); p=read_json(out/"protocol.json"); gt=read_json(p["inputs"]["annotations"])
    predictions={k:read_json(v) for k,v in p["inputs"]["predictions"].items()}
    behavior(gt,predictions,out/"behavior")
    vectors={}; sensitivity={}; subset=read_json(p["inputs"]["sensitivity_annotations"]); ids={i["id"] for i in subset["images"]}
    for label,rows in predictions.items():
        cache=CocoCache(gt,rows); vectors[label]=cache.reference.tolist()
        sensitivity[label]=metrics(cache.accumulate(np.array([int(i) in ids for i in cache.ids],dtype=int)))
    write_json(out/"point_ap_vectors.json",vectors); write_json(out/"duplicate_sensitivity.json",sensitivity)
    rules=p["rules"]
    bootstrap(gt,predictions,out/"uncertainty",rules["bootstrap_replicates"],rules["bootstrap_seed"],rules["cpu_workers"])


def execute(output):
    out=Path(output); p=read_json(out/"protocol.json"); source=Path(p["source_root"])
    state={"status":"running","pipeline_pid":os.getpid(),"completed":[]}; stats=None
    def save():
        state["updated_at"]=dt.datetime.now(dt.timezone.utc).isoformat(); write_json(out/"state.json",state)
    def verify():
        for rel,digest in p["source_sha256"].items():
            if sha256(source/rel)!=digest: raise RuntimeError(f"Frozen analysis code changed: {rel}")
        for path,digest in p["input_sha256"].items():
            if sha256(path)!=digest: raise RuntimeError(f"Frozen input changed: {path}")
    def command(label,argv):
        verify(); state["stage"]=label; save()
        with (out/f"{label}.log").open("w",encoding="utf-8") as log:
            child=subprocess.Popen([sys.executable,*argv],cwd=source,stdout=log,stderr=subprocess.STDOUT)
            state["child_pid"]=child.pid; save(); code=child.wait()
        if code: raise RuntimeError(f"{label} exited {code}; see {out/f'{label}.log'}")
        state["completed"].append(label); save()
    save()
    try:
        command("unit_checks",["-m","unittest","discover","-s","tests","-p","test_mssl_analysis.py","-v"])
        from .verify_math import run as math
        math(p["workspace"],out/"math_verification.json"); state["completed"].append("math_verification"); save()
        # Verify raw image bytes against the preexisting frozen manifest, not GT masks.
        images={im["id"] for im in read_json(p["inputs"]["annotations"])["images"]}
        manifest=read_json(p["inputs"]["image_manifest"])
        # Manifest records have no numeric ID; compare the scored relative paths.
        names={im["file_name"] for im in read_json(p["inputs"]["annotations"])["images"]}
        for image in manifest:
            if image["relative_path"] in names and sha256(Path(p["image_root"])/image["relative_path"])!=image["sha256"]:
                raise RuntimeError("Target image changed")
        with (out/"statistics.log").open("w",encoding="utf-8") as log:
            stats=subprocess.Popen([sys.executable,"-m","polypbench.analysis.study","cpu","--output",str(out)],
                cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state["statistics_pid"]=stats.pid; save()
        for label,training in p["training"].items():
            job={**training,"annotations":p["inputs"]["annotations"],"predictions":p["inputs"]["predictions"][label],
                 "image_root":p["image_root"],"output":str(out/"score_replay"/label)}
            jobpath=out/f"{label}_replay_job.json"; write_json(jobpath,job)
            command(f"{label}_score_replay",["-m","polypbench.analysis.score_replay","--job",str(jobpath)])
            if stats.poll() not in (None,0): raise RuntimeError("CPU statistics failed; see statistics.log")
        state["stage"]="waiting_for_statistics"; save()
        if stats.wait(): raise RuntimeError("CPU statistics failed; see statistics.log")
        state["completed"].append("statistics"); save()
        verify()
        from .score_effects import run as scores
        gt=read_json(p["inputs"]["annotations"]); predictions={k:read_json(v) for k,v in p["inputs"]["predictions"].items()}
        state["stage"]="score_counterfactuals"; save(); scores(gt,predictions,out/"score_replay",out/"score_effects")
        state["completed"].append("score_counterfactuals"); save()
        from .report import run as report
        state["stage"]="report"; save(); report(out)
        state["completed"].append("report"); state["status"]="completed"
    except BaseException as error:
        state.update(status="failed",error=repr(error))
        if stats is not None and stats.poll() is None:
            if os.getpgid(stats.pid)==stats.pid: os.killpg(stats.pid,signal.SIGTERM)
        raise
    finally: save()


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=("launch","run","cpu")); parser.add_argument("--output",required=True)
    args=parser.parse_args(); {"launch":prepare,"run":execute,"cpu":cpu}[args.action](args.output)
