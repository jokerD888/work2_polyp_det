"""Freeze a campaign source bundle and start its local pipeline detached."""
import argparse
import copy
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml
from polypbench.io import read_json,sha256,write_json
from .config import load_experiment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--precheck",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--reuse-smoke-run",help="Reuse only a completed, same-code/config/data smoke")
    args=parser.parse_args()
    report=read_json(args.precheck)
    if report.get("status")!="passed":
        raise ValueError("A passed real-GPU acceptance report is required")
    cfg=load_experiment(args.config)
    root=Path(cfg["project_root"])
    if args.reuse_smoke_run:
        from .acceptance import smoke_gate
        from .run import validate_dataset
        old=Path(cfg["output_root"])/args.reuse_smoke_run
        smoke_gate(old)
        old_sources=read_json(old/"adapter_sources.json")
        critical=("config.py","loss.py","model.py","data.py","worker.py","accumulation.py","run.py")
        for name in critical:
            relative=f"methods/mssl/{name}"
            if old_sources.get(relative)!=sha256(root/relative):
                raise ValueError(f"Cannot reuse smoke after training-code change: {relative}")
        old_cfg=read_json(old/"experiment.json")
        for key in ("dataset","model","mssl","runtime","seed","repo"):
            if old_cfg[key]!=cfg[key]:
                raise ValueError(f"Cannot reuse smoke with changed {key}")
        expected_overrides=dict(cfg["overrides"])
        expected_overrides.update({"SOLVER.MAX_ITER":16,"SOLVER.STEPS":[],
                                  "SOLVER.CHECKPOINT_PERIOD":16,"TEST.EVAL_PERIOD":16})
        if old_cfg["overrides"]!=expected_overrides:
            raise ValueError("Cannot reuse smoke with different training settings")
        if read_json(old/"manifest.json")["dataset"]!=validate_dataset(cfg):
            raise ValueError("Cannot reuse smoke after dataset change")
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=output/"source";source.mkdir()
    ignore=shutil.ignore_patterns("__pycache__","*.pyc")
    shutil.copytree(root/"polypbench",source/"polypbench",ignore=ignore)
    (source/"methods").mkdir()
    shutil.copy2(root/"methods/__init__.py",source/"methods/__init__.py")
    shutil.copytree(root/"methods/mssl",source/"methods/mssl",ignore=ignore)
    (source/"configs").mkdir()
    frozen=copy.deepcopy(cfg)
    frozen["project_root"]=".."
    frozen.pop("source_config",None)
    config=source/"configs/formal.yaml"
    config.write_text(yaml.safe_dump(frozen,sort_keys=False),encoding="utf-8")
    smoke=copy.deepcopy(frozen)
    smoke["name"]+="_healthcheck"
    smoke["purpose"]="healthcheck_not_performance_evidence"
    smoke.pop("training_budget",None)
    smoke["overrides"].update({"SOLVER.MAX_ITER":16,"SOLVER.STEPS":[],
                              "SOLVER.CHECKPOINT_PERIOD":16,"TEST.EVAL_PERIOD":16})
    (source/"configs/smoke.yaml").write_text(yaml.safe_dump(smoke,sort_keys=False),encoding="utf-8")
    stamp=dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    metadata={"created_at":dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root":str(root),"source_root":str(source),"output_root":cfg["output_root"],
        "precheck":str(Path(args.precheck).resolve()),"precheck_sha256":sha256(args.precheck),
        "source_sha256":{p.relative_to(source).as_posix():sha256(p) for p in source.rglob("*") if p.is_file()},
        "runs":{"smoke":f"mssl-smoke-{stamp}","method":f"mssl-official50-s42-{stamp}",
                "control":f"maskdino-mssl-control-s42-{stamp}"},
        "order":["smoke_train","smoke_predict","smoke_health_gate","method_train","method_predict",
                 "control_train","control_predict"],"status":"starting"}
    write_json(output/"campaign.json",metadata)
    if args.reuse_smoke_run:
        metadata["reuse_smoke_run"]=args.reuse_smoke_run
        metadata["runs"]["smoke"]=args.reuse_smoke_run
        write_json(output/"campaign.json",metadata)
    env=os.environ.copy();env["PYTHONPATH"]=str(source)+os.pathsep+env.get("PYTHONPATH","")
    env["PYTHONUNBUFFERED"]="1"
    with (output/"pipeline.log").open("w",encoding="utf-8") as log:
        child=subprocess.Popen([sys.executable,"-m","methods.mssl.pipeline","--campaign",str(output)],
            cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
    write_json(output/"launcher.json",{"pid":child.pid,"campaign":str(output),
                                       "log":str(output/"pipeline.log")})
    print(f"Detached campaign PID {child.pid}: {output}",flush=True)


if __name__=="__main__":
    main()
