"""Finite detached acceptance -> smoke -> one full run -> five-group comparison."""
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml
from polypbench.io import read_json, write_json, sha256


def launch(output):
    from methods.mssl.config import load_experiment
    from methods.mssl.run import verify_repo, validate_dataset
    from .config import validate
    from .loss import reference_record
    from polypbench.zeroshot import validate_frozen_training
    workspace=Path.cwd().resolve()
    prior=read_json(workspace/"runs/mssl-attribution-20261018-r2/protocol.json")
    if read_json(workspace/"runs/mssl-polypgen-analysis-20261019/state.json")["status"]!="completed":
        raise RuntimeError("Prior four-model study incomplete")
    original=read_json(workspace/"runs"/prior["runs"]["baseline"]/"experiment.json")
    cfg=validate(load_experiment(workspace/"methods/ciou_replace/configs/kvasir_official50.yaml"))
    common=("dataset","overrides","model","runtime","seed","checkpoint_selection","checkpoint_keep","training_budget","repo","evaluation")
    for key in common:
        if cfg[key]!=original[key]: raise RuntimeError(f"Not the original training protocol: {key}")
    dataset=validate_dataset(cfg)
    if dataset!=read_json(workspace/"runs"/prior["runs"]["baseline"]/"manifest.json")["dataset"]:
        raise RuntimeError("Original source data changed")
    if verify_repo(cfg)!=read_json(workspace/"runs"/prior["runs"]["baseline"]/"manifest.json")["repo"]:
        raise RuntimeError("Original native code changed")
    references={}
    for label,run_id in prior["runs"].items():
        trained=workspace/"runs"/run_id; validate_frozen_training(trained)
        for rel in ("experiment.json","manifest.json","train/selected_checkpoint.json","train/model_best.pth","predict_val/predictions.bbox.json"):
            references[str(trained/rel)]=sha256(trained/rel)
    previous_analysis=workspace/"runs/mssl-polypgen-analysis-20261019"
    analysis=read_json(previous_analysis/"protocol.json")
    for path in analysis["inputs"]["predictions"].values(): references[path]=sha256(path)
    for rel in ("protocol.json","uncertainty/image_multiplicities.npy","uncertainty/bootstrap_ap_vectors.npz"):
        references[str(previous_analysis/rel)]=sha256(previous_analysis/rel)
    target=Path(analysis["inputs"]["annotations"]).parent
    for path in target.glob("*.json"): references[str(path)]=sha256(path)
    out=Path(output).resolve(); out.mkdir(parents=True,exist_ok=False)
    source=out/"source"; source.mkdir()
    ignore=shutil.ignore_patterns("__pycache__","*.pyc")
    shutil.copytree(workspace/"polypbench",source/"polypbench",ignore=ignore)
    (source/"methods").mkdir(); shutil.copy2(workspace/"methods/__init__.py",source/"methods/__init__.py")
    for name in ("mssl","ciou_replace"): shutil.copytree(workspace/"methods"/name,source/"methods"/name,ignore=ignore)
    (source/"tests").mkdir()
    shutil.copy2(workspace/"tests/test_mssl_analysis.py",source/"tests/test_mssl_analysis.py")
    (source/"configs").mkdir()
    formal=copy.deepcopy(cfg); formal.pop("source_config",None); formal["project_root"]=".."
    formal_path=source/"configs/formal.yaml"; formal_path.write_text(yaml.safe_dump(formal,sort_keys=False),encoding="utf-8")
    smoke=copy.deepcopy(formal); smoke["name"]+="_smoke"; smoke["purpose"]="healthcheck_not_method_selection"
    smoke.pop("training_budget",None)
    smoke["overrides"].update({"SOLVER.MAX_ITER":50,"SOLVER.STEPS":[],"SOLVER.CHECKPOINT_PERIOD":50,"TEST.EVAL_PERIOD":50})
    smoke_path=source/"configs/smoke.yaml"; smoke_path.write_text(yaml.safe_dump(smoke,sort_keys=False),encoding="utf-8")
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs={**prior["runs"],"ciou_replace":f"ciou-replace-s42-{stamp}"}
    protocol={"created_at":dt.datetime.now(dt.timezone.utc).isoformat(),"workspace":str(workspace),"source_root":str(source),
        "runs":runs,"smoke_run":f"ciou-smoke-{stamp}","formal_config":str(formal_path),"smoke_config":str(smoke_path),
        "new_complete_training_runs":1,"common_protocol_fields_verified_equal":list(common),
        "reference_ciou":reference_record(),"dataset_fingerprint":dataset,
        "ciou_contract":cfg["box_regression"],"regression_weight":2.,"l1_weight":5.,
        "reference_input_sha256":references,"image_root":analysis["image_root"],"annotations":analysis["inputs"]["annotations"],
        "image_manifest":analysis["inputs"]["image_manifest"],"target_root":str(target),
        "source_sha256":{p.relative_to(source).as_posix():sha256(p) for p in source.rglob("*") if p.is_file()},
        "evaluation":"source max AP-selected weights, original fixed1473 target; all5 share original2000 paired draws",
        "deferred":["V2","multi-seed","PolypsSet","PolypDB","PICCOLO"],
        "order":["ciou_unit_checks","existing_mssl_regression_checks","bootstrap_regression_checks","precheck",
                 "smoke_train","smoke_predict","smoke_gate","formal_train","formal_predict","checkpoint_replay",
                 "polypgen_inference","polypgen_scoring","five_group_comparison"]}
    write_json(out/"protocol.json",protocol)
    env=os.environ.copy(); env["PYTHONPATH"]=str(source)+os.pathsep+env.get("PYTHONPATH",""); env["PYTHONUNBUFFERED"]="1"
    with (out/"pipeline.log").open("w",encoding="utf-8") as log:
        child=subprocess.Popen([sys.executable,"-m","methods.ciou_replace.campaign","run","--output",str(out)],
            cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/"launcher.json",{"pid":child.pid,"output":str(out),"detached":True})
    print(f"Detached CIoU campaign PID {child.pid}: {out}",flush=True)


def smoke_gate(training):
    path=Path(training)
    for stage in ("train","predict_val"):
        if read_json(path/stage/"state.json")["status"]!="completed": raise RuntimeError("Incomplete smoke stage")
    rows=[json.loads(line) for line in (path/"train/metrics.json").read_text().splitlines() if line.strip()]
    rows=[row for row in rows if "optimizer/successful_updates" in row and "total_loss" in row]
    last=max(rows,key=lambda r:r["iteration"])
    if last["iteration"]<49 or last["optimizer/successful_updates"]<25:
        raise RuntimeError("Smoke lacks sufficient successful updates after AMP backoff")
    if any("loss_mssl" in row for row in rows): raise RuntimeError("Unexpected MSSL loss in CIoU smoke")
    read_json(path/"predict_val/metrics.json")
    return {"status":"passed","last_logged_iteration":last["iteration"],"successful_updates":last["optimizer/successful_updates"],
        "loss_scale":last["amp/loss_scale"],"full_val_and_checkpoint_reload":True,"scope":"numerical/runtime only, never AP-based selection"}


def execute(output):
    out=Path(output); p=read_json(out/"protocol.json"); workspace=Path(p["workspace"]); source=Path(p["source_root"])
    state={"status":"running","pipeline_pid":os.getpid(),"completed":[]}
    def save():
        state["updated_at"]=dt.datetime.now(dt.timezone.utc).isoformat(); write_json(out/"state.json",state)
    def verify():
        for name,digest in p["source_sha256"].items():
            if sha256(source/name)!=digest: raise RuntimeError(f"Frozen source changed: {name}")
        for path,digest in p["reference_input_sha256"].items():
            if sha256(path)!=digest: raise RuntimeError(f"Reference changed: {path}")
    def command(label,args):
        verify(); state.update(stage=label,command=[sys.executable,*args]); save()
        with (out/f"{label}.log").open("w",encoding="utf-8") as log:
            child=subprocess.Popen([sys.executable,*args],cwd=source,stdout=log,stderr=subprocess.STDOUT)
            state["child_pid"]=child.pid; save(); code=child.wait()
        if code: raise RuntimeError(f"{label} failed ({code}); see {out/f'{label}.log'}")
        state["completed"].append(label); save()
    def recorded(label):
        state["completed"].append(label); save()
    save()
    try:
        command("ciou_unit_checks",["-m","unittest","discover","-s","methods/ciou_replace/tests","-v"])
        command("existing_mssl_regression_checks",["-m","unittest","discover","-s","methods/mssl/tests","-v"])
        command("bootstrap_regression_checks",["-m","unittest","discover","-s","tests","-p","test_mssl_analysis.py","-v"])
        command("precheck",["-m","methods.ciou_replace.precheck","--config",p["formal_config"],"--output",str(out/"precheck"),"--panel-size","2"])
        if read_json(out/"precheck/report.json")["status"]!="passed": raise RuntimeError("GPU precheck did not pass")
        smoke=p["smoke_run"]
        for action in ("train","predict"):
            command(f"smoke_{action}",["-m","methods.ciou_replace",action,"--config",p["smoke_config"],"--run-id",smoke])
        state["stage"]="smoke_gate"; save()
        write_json(out/"smoke_gate.json",smoke_gate(workspace/"runs"/smoke)); recorded("smoke_gate")
        trained=workspace/"runs"/p["runs"]["ciou_replace"]
        for action in ("train","predict"):
            command(f"formal_{action}",["-m","methods.ciou_replace",action,"--config",p["formal_config"],"--run-id",p["runs"]["ciou_replace"]])
        command("checkpoint_replay",["-m","methods.ciou_replace.precheck","--config",p["formal_config"],"--training",str(trained),
                                      "--output",str(out/"checkpoint_replay"),"--panel-size","4"])
        verify(); state["stage"]="validate_target_images"; save()
        gt=read_json(p["annotations"]); names={im["file_name"] for im in gt["images"]}
        for image in read_json(p["image_manifest"]):
            if image["relative_path"] in names and sha256(Path(p["image_root"])/image["relative_path"])!=image["sha256"]:
                raise RuntimeError("Target image bytes changed")
        from polypbench.zeroshot import validate_frozen_training, score_cohort, COHORTS
        _,checkpoint,digest=validate_frozen_training(trained)
        external=out/"external"; external.mkdir(exist_ok=False)
        write_json(external/"primary.images.json",{"images":gt["images"]})
        write_json(external/"job.json",{"experiment":str(trained/"experiment.json"),"checkpoint":str(checkpoint),"checkpoint_sha256":digest,
            "output":str(external),"image_root":p["image_root"],"all_images":str(external/"primary.images.json"),
            "source_predictions":str(trained/"predict_val/predictions.bbox.json")})
        command("polypgen_inference",["-m","polypbench.adapters.zeroshot_worker","--job",str(external/"job.json")])
        state["stage"]="polypgen_scoring"; save()
        predictions=read_json(external/"predictions.all_stills.bbox.json")
        for name,filename in COHORTS.items(): score_cohort(Path(p["target_root"])/filename,predictions,external/name)
        recorded("polypgen_scoring")
        command("five_group_comparison",["-m","methods.ciou_replace.comparison","--output",str(out)])
        state["status"]="completed"
    except BaseException as error:
        state.update(status="failed",error=repr(error)); raise
    finally:
        save()


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=("launch","run")); parser.add_argument("--output",required=True)
    args=parser.parse_args(); (launch if args.action=="launch" else execute)(args.output)
