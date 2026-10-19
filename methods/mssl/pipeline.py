"""Smoke -> method -> matched control; fail closed on errors or source drift."""
import argparse
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys

from polypbench.io import read_json,sha256,write_json
from .acceptance import smoke_gate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--campaign",required=True)
    args=parser.parse_args()
    campaign=Path(args.campaign)
    record=read_json(campaign/"campaign.json")
    source=Path(record["source_root"])
    state={"status":"running","pipeline_pid":os.getpid(),"completed":[]}
    def save():
        state["updated_at"]=dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(campaign/"state.json",state)
    def run(label,action,config,run_id,baseline=False):
        for name,digest in record["source_sha256"].items():
            if sha256(source/name)!=digest:
                raise RuntimeError(f"Frozen source changed: {name}")
        cmd=[sys.executable,"-m","methods.mssl",action,"--config",str(config),"--run-id",run_id]
        if baseline:cmd.append("--baseline")
        state.update(stage=label,run_id=run_id,command=cmd)
        save()
        print(f"\nSTAGE {label}: {run_id}",flush=True)
        with (campaign/f"{label}.log").open("w",encoding="utf-8") as log:
            child=subprocess.Popen(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT)
            state["child_pid"]=child.pid;save()
            code=child.wait()
        if code:
            raise RuntimeError(f"{label} exited {code}; see {campaign/f'{label}.log'}")
        state["completed"].append(label);save()
    save()
    try:
        runs=record["runs"]
        if not record.get("reuse_smoke_run"):
            run("smoke_train","train",source/"configs/smoke.yaml",runs["smoke"])
            run("smoke_predict","predict",source/"configs/smoke.yaml",runs["smoke"])
        else:
            state["completed"].extend(["smoke_train_reused","smoke_predict_reused"])
        root=Path(record["output_root"])/runs["smoke"]
        gate=smoke_gate(root)
        write_json(campaign/"smoke_gate.json",gate)
        state["completed"].append("smoke_health_gate");save()
        for group,baseline in (("method",False),("control",True)):
            for action in ("train","predict"):
                run(f"{group}_{action}",action,source/"configs/formal.yaml",runs[group],baseline)
        state["status"]="completed"
    except BaseException as error:
        state.update(status="failed",error=repr(error))
        raise
    finally:
        save()


if __name__=="__main__":
    main()
