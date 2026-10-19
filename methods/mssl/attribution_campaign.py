"""Frozen sequential attribution campaign, with explicit state and fail-closed gates."""
import argparse
import copy
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from polypbench.io import read_json, sha256, write_json
from .config import load_experiment, MSSLOptions

OLD={"baseline":"maskdino-mssl-control-s42-20261018T025946",
     "mssl":"mssl-official50-s42-20261018T025946"}


def launch(output):
    root=Path.cwd().resolve()
    original=read_json(root/"runs"/OLD["mssl"]/"experiment.json")
    baseline=read_json(root/"runs"/OLD["baseline"]/"experiment.json")
    for key in ("dataset","overrides","model","runtime","seed","checkpoint_selection","training_budget","repo"):
        if original[key]!=baseline[key]: raise RuntimeError(f"Original paired protocol mismatch: {key}")
    from .run import validate_dataset, verify_repo
    if validate_dataset(original)!=read_json(root/"runs"/OLD["mssl"]/"manifest.json")["dataset"]:
        raise RuntimeError("Original training data changed")
    if verify_repo(original)!=read_json(root/"runs"/OLD["mssl"]/"manifest.json")["repo"]:
        raise RuntimeError("Native implementation changed")
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    source=output/"source"; source.mkdir()
    ignore=shutil.ignore_patterns("__pycache__","*.pyc")
    shutil.copytree(root/"polypbench",source/"polypbench",ignore=ignore)
    (source/"methods").mkdir(); shutil.copy2(root/"methods/__init__.py",source/"methods/__init__.py")
    shutil.copytree(root/"methods/mssl",source/"methods/mssl",ignore=ignore)
    (source/"configs").mkdir()
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs={**OLD,"box_cdf":f"mssl-box-cdf-s42-{stamp}","mask_static":f"mssl-mask-static-s42-{stamp}"}
    configurations={}
    for label,support,form in (("box_cdf","box","cdf"),("mask_static","mask","static")):
        cfg=copy.deepcopy(original)
        cfg.pop("source_config",None)
        cfg["project_root"]=".."
        cfg["name"]=f"mssl_{label}_official50"
        cfg["purpose"]="prespecified_mechanism_attribution_no_parameter_search"
        cfg["gradient_probe_period"]=275
        cfg["mssl"]={**MSSLOptions().__dict__,**cfg["mssl"],"support":support,"loss_form":form}
        path=source/"configs"/f"{label}.yaml"
        path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
        loaded=load_experiment(path)
        for key in ("dataset","overrides","model","runtime","seed","checkpoint_selection","training_budget","repo"):
            if loaded[key]!=original[key]: raise RuntimeError(f"Unexpected protocol change: {label}/{key}")
        configurations[label]=str(path)
    target=root/"data/polypgen_zeroshot_stills_v2"
    metadata={"created_at":dt.datetime.now(dt.timezone.utc).isoformat(),"workspace":str(root),
        "source_root":str(source),"runs":runs,"configs":configurations,
        "source_sha256":{p.relative_to(source).as_posix():sha256(p) for p in source.rglob("*") if p.is_file()},
        "reference_run_sha256":{label:{name:sha256(root/"runs"/run/name) for name in
            ("experiment.json","manifest.json","train/selected_checkpoint.json","predict_val/predictions.bbox.json")}
            for label,run in OLD.items()},
        "target_data_root":str(target),"target_data_sha256":{p.name:sha256(p) for p in target.glob("*.json")},
        "rules":{"seed":42,"epochs":50,"new_formal_training_runs":2,"multi_seed":False,
            "weight":1.,"temperature":.05,"bandwidth":.15,"selection":"best source val bbox AP, earliest tie",
            "no_target_tuning":True,"static":"0.5*Gprime(0)^2*d^2; original losses retained",
            "box_control":"replace support only; original mask supervision/eligibility retained",
            "gradient_probes":"fixed first microbatch at iterations 0,10,275*k; no backward accumulation or RNG change",
            "interpretation":"single seed attribution; associations exploratory; equality not determined by AP rounding"}}
    write_json(output/"protocol.json",metadata)
    env=os.environ.copy(); env["PYTHONPATH"]=str(source)+os.pathsep+env.get("PYTHONPATH",""); env["PYTHONUNBUFFERED"]="1"
    with (output/"pipeline.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,"-m","methods.mssl.attribution_campaign","run","--output",str(output)],
            cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(output/"launcher.json",{"pid":process.pid,"detached":True,"output":str(output)})
    print(f"Detached attribution campaign PID {process.pid}: {output}",flush=True)


def run(output):
    out=Path(output); protocol=read_json(out/"protocol.json"); root=Path(protocol["workspace"]); source=Path(protocol["source_root"])
    state={"status":"running","pipeline_pid":os.getpid(),"completed":[]}
    def save():
        state["updated_at"]=dt.datetime.now(dt.timezone.utc).isoformat(); write_json(out/"state.json",state)
    def verify():
        for path,digest in protocol["source_sha256"].items():
            if sha256(source/path)!=digest: raise RuntimeError(f"Frozen source drift: {path}")
        for label,records in protocol["reference_run_sha256"].items():
            for name,digest in records.items():
                if sha256(root/"runs"/protocol["runs"][label]/name)!=digest: raise RuntimeError("Reference run changed")
    def command(label,args):
        verify(); state.update(stage=label,command=[sys.executable,*args]); save()
        print(f"STAGE {label}",flush=True)
        with (out/f"{label}.log").open("w",encoding="utf-8") as log:
            child=subprocess.Popen([sys.executable,*args],cwd=source,stdout=log,stderr=subprocess.STDOUT)
            state["child_pid"]=child.pid; save(); code=child.wait()
        if code: raise RuntimeError(f"{label} exited {code}; see {out/f'{label}.log'}")
        state["completed"].append(label); save()
    def diagnostics(label,group_names,external=False):
        from .diagnostics import run as diagnose
        mapping={g:str(root/"runs"/protocol["runs"][g]/"predict_val/predictions.bbox.json") for g in group_names}
        cfg=read_json(root/"runs"/OLD["mssl"]/"experiment.json")
        annotation=cfg["dataset"]["val"]["annotations"]; mask_root=cfg["dataset"]["raw_mask_root"]
        if external:
            mapping={g:str((root/"runs/polypgen-zeroshot-20261018"/g/"pooled/predictions.bbox.json")
                           if g in OLD else (out/"external"/g/"pooled/predictions.bbox.json")) for g in group_names}
            annotation=str(Path(protocol["target_data_root"])/"provided_boxes.coco.json")
            mask_root=read_json(root/"runs/polypgen-zeroshot-20261018/protocol.json")["image_root"]
        state["stage"]=label; save(); diagnose(annotation,mapping,out/label,mask_root,"polypgen" if external else "source")
        state["completed"].append(label); save()
    save()
    try:
        command("unit_checks",["-m","unittest","discover","-s","methods/mssl/tests","-v"])
        command("loss_curves",["-m","methods.mssl.curves","--config",protocol["configs"]["box_cdf"],"--output",str(out/"loss_curves")])
        command("box_cdf_precheck",["-m","methods.mssl.precheck","--config",protocol["configs"]["box_cdf"],"--output",str(out/"box_cdf_precheck")])
        # Prediction analysis uses CPU only; complete before long training so an
        # actionable preliminary report is available without waiting for new runs.
        diagnostics("existing_source_diagnostics",list(OLD))
        diagnostics("existing_polypgen_diagnostics",list(OLD),True)
        for group in ("box_cdf","mask_static"):
            if group=="mask_static":
                # Real-data, full AMP backward plus inference parity before static training.
                command("mask_static_precheck",["-m","methods.mssl.precheck","--config",protocol["configs"][group],"--output",str(out/"mask_static_precheck")])
            for action in ("train","predict"):
                command(f"{group}_{action}",["-m","methods.mssl",action,"--config",protocol["configs"][group],"--run-id",protocol["runs"][group]])
            if group=="box_cdf":
                for label in ("baseline","mssl","box_cdf"):
                    command(f"{label}_replay",["-m","methods.mssl.replay","--training",str(root/"runs"/protocol["runs"][label]),"--output",str(out/f"{label}_replay")])
        command("mask_static_replay",["-m","methods.mssl.replay","--training",str(root/"runs"/protocol["runs"]["mask_static"]),"--output",str(out/"mask_static_replay")])
        from polypbench.zeroshot import COHORTS, score_cohort, validate_frozen_training
        for name,digest in protocol["target_data_sha256"].items():
            if sha256(Path(protocol["target_data_root"])/name)!=digest: raise RuntimeError("Target cohort drift")
        for group in ("box_cdf","mask_static"):
            training=root/"runs"/protocol["runs"][group]
            cfg,checkpoint,digest=validate_frozen_training(training)
            dest=out/"external"/group; dest.mkdir(parents=True,exist_ok=False)
            old_protocol=read_json(root/"runs/polypgen-zeroshot-20261018/protocol.json")
            # The target image manifest and exact source-selected checkpoint are
            # fixed; inference receives images only, never target annotations.
            manifest=read_json(Path(protocol["target_data_root"])/"image_manifest.json")
            for image in manifest:
                if sha256(Path(old_protocol["image_root"])/image["relative_path"])!=image["sha256"]:
                    raise RuntimeError("Target image drift")
            job={"experiment":str(training/"experiment.json"),"checkpoint":str(checkpoint),"checkpoint_sha256":digest,
                 "output":str(dest),"image_root":old_protocol["image_root"],
                 "all_images":str(Path(protocol["target_data_root"])/"all_stills.images.json"),
                 "source_predictions":str(training/"predict_val/predictions.bbox.json")}
            write_json(dest/"job.json",job)
            command(f"{group}_external_inference",["-m","polypbench.adapters.zeroshot_worker","--job",str(dest/"job.json")])
            predictions=read_json(dest/"predictions.all_stills.bbox.json")
            for name,filename in COHORTS.items(): score_cohort(Path(protocol["target_data_root"])/filename,predictions,dest/name)
        diagnostics("four_group_source_diagnostics",list(protocol["runs"]))
        diagnostics("four_group_polypgen_diagnostics",list(protocol["runs"]),True)
        from .attribution_report import run as report
        report(out)
        state["completed"].append("attribution_report"); state["status"]="completed"
    except BaseException as error:
        state.update(status="failed",error=repr(error)); raise
    finally: save()


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=("launch","run")); parser.add_argument("--output",required=True)
    args=parser.parse_args(); (launch if args.action=="launch" else run)(args.output)
