"""Frozen Kvasir -> PolypGen evaluation, with no target-dependent selection."""
import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .io import read_json,sha256,write_json
from .evaluation import evaluate
from .runner import git_info


RUNS={"baseline":"maskdino-mssl-control-s42-20261018T025946",
      "mssl":"mssl-official50-s42-20261018T025946"}
COHORTS={"pooled":"provided_boxes.coco.json",**{f"C{i}":f"C{i}.coco.json" for i in range(1,7)},
         "without_internal_duplicates":"without_internal_duplicates.coco.json"}


def validate_frozen_training(run_root):
    experiment=read_json(run_root/"experiment.json")
    recorded=read_json(run_root/"adapter_sources.json")
    for file,digest in recorded.items():
        if sha256(Path(experiment["project_root"])/file)!=digest:
            raise RuntimeError(f"Trained source changed: {file}")
    if git_info(experiment["repo"]["path"])!=read_json(run_root/"manifest.json")["repo"]:
        raise RuntimeError("Pinned native implementation changed since training")
    checkpoint=run_root/"train/model_best.pth"
    selected=read_json(run_root/"train/selected_checkpoint.json")
    if sha256(checkpoint)!=selected["sha256"]:
        raise RuntimeError("Source-selected checkpoint changed")
    return experiment,checkpoint,selected["sha256"]


def score_cohort(annotations, predictions, output):
    gt=read_json(annotations);ids={x["id"] for x in gt["images"]}
    rows=[row for row in predictions if row["image_id"] in ids]
    write_json(output/"predictions.bbox.json",rows)
    metric=evaluate(annotations,output/"predictions.bbox.json",output/"metrics.json")
    positive={x["image_id"] for x in gt["annotations"]};negative=ids-positive
    counts={i:0 for i in negative}
    for row in rows:
        if row["image_id"] in counts and row["score"]>=.5:
            counts[row["image_id"]]+=1
    write_json(output/"negative_frame_diagnostic.json",{
        "score_threshold":.5,"threshold_policy":"fixed before target inference; diagnostic only",
        "negative_images":len(negative),"false_positive_detections":sum(counts.values()),
        "negative_frames_with_detection":sum(v>0 for v in counts.values()),
        "false_positive_detections_per_negative_frame":sum(counts.values())/len(negative) if negative else None,
        "negative_frame_false_alarm_fraction":sum(v>0 for v in counts.values())/len(negative) if negative else None,
        "COCO_AP_uses_all_scores":True})
    return metric


def run(output):
    root=Path(output);protocol=read_json(root/"protocol.json")
    state={"status":"running","stage":"validation","completed":[]}
    def save():
        state["updated_at"]=dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(root/"state.json",state)
    save()
    try:
        for relative,digest in protocol["evaluation_source_sha256"].items():
            if sha256(Path(protocol["project_root"])/relative)!=digest:
                raise RuntimeError("External evaluation code changed after protocol freeze")
        for filename,digest in protocol["data_sha256"].items():
            if sha256(root/"data"/filename)!=digest:
                raise RuntimeError("Frozen evaluation cohort changed")
        manifest=read_json(root/"data/image_manifest.json")
        for image in manifest:
            if sha256(Path(protocol["image_root"])/image["relative_path"])!=image["sha256"]:
                raise RuntimeError(f"Target image changed: {image['relative_path']}")
        result_rows=[];results={}
        for group,run_id in RUNS.items():
            for relative,digest in protocol["evaluation_source_sha256"].items():
                if sha256(Path(protocol["project_root"])/relative)!=digest:
                    raise RuntimeError("External evaluation code changed between models")
            training=Path(protocol["project_root"])/"runs"/run_id
            experiment,checkpoint,digest=validate_frozen_training(training)
            group_out=root/group;group_out.mkdir(exist_ok=False)
            job={"experiment":str(training/"experiment.json"),"checkpoint":str(checkpoint),
                 "checkpoint_sha256":digest,"output":str(group_out),"image_root":protocol["image_root"],
                 "all_images":str(root/"data/all_stills.images.json"),
                 "source_predictions":str(training/"predict_val/predictions.bbox.json")}
            write_json(group_out/"job.json",job)
            state.update(stage=f"{group}_inference",run_id=run_id);save()
            command=[sys.executable,"-m","polypbench.adapters.zeroshot_worker","--job",str(group_out/"job.json")]
            with (group_out/"console.log").open("w",encoding="utf-8") as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=protocol["project_root"])
                state["child_pid"]=process.pid;save()
                code=process.wait()
            if code:
                raise RuntimeError(f"{group} inference exited {code}; see {group_out/'console.log'}")
            predictions=read_json(group_out/"predictions.all_stills.bbox.json")
            state["stage"]=f"{group}_scoring";save()
            results[group]={}
            for name,filename in COHORTS.items():
                scored=score_cohort(root/"data"/filename,predictions,group_out/name)
                results[group][name]=scored["metrics"]
                result_rows.append({"model":group,"cohort":name,"images":scored["images"],
                    **{k:(100*v if v is not None else None) for k,v in scored["metrics"].items()}})
            state["completed"].append(group);save()
        deltas={name:{key:100*(results["mssl"][name][key]-results["baseline"][name][key])
                         if results["mssl"][name][key] is not None and results["baseline"][name][key] is not None else None
                         for key in results["mssl"][name]} for name in COHORTS}
        write_json(root/"summary.json",{"results_fraction":results,"mssl_minus_baseline_percentage_points":deltas,
            "protocol":protocol,"scope":"single seed, no PolypGen training or tuning, source-validation-selected weights"})
        with (root/"results_percent.csv").open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(result_rows[0]));writer.writeheader();writer.writerows(result_rows)
        lines=["# Kvasir → PolypGen 零样本检测评估","",
               "两组均使用 Kvasir val 选定的最佳权重；没有目标域训练、微调、阈值搜索或权重选择。", "",
               "原始单帧 1537 张全部推理；主评估 1473 张有框文件图像、1533 个框，含 126 张原始空框负样本。",
               "C3 另 64 张缺框，仅保存预测；不伪造为负样本。敏感性评估排除 C5 两对像素相同但标注不一致的图像，共 1469 张。", "",
               "| 范围 | 基线 AP | MSSL AP | ΔAP | 基线 AP75 | MSSL AP75 | ΔAP75 |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for name in COHORTS:
            b,m=results["baseline"][name],results["mssl"][name]
            lines.append(f"| {name} | {100*b['AP']:.2f} | {100*m['AP']:.2f} | {deltas[name]['AP']:+.2f} | {100*b['AP75']:.2f} | {100*m['AP75']:.2f} | {deltas[name]['AP75']:+.2f} |")
        for group in RUNS:
            macro=sum(results[group][f"C{i}"]["AP"] for i in range(1,7))/6
            lines+= ["",f"{group} 六中心 AP 算术平均：{macro*100:.2f}（与 pooled COCO AP 定义不同）。"]
        lines +=["","评估：原生 800/1333 resize、100 detections、原生 Mask 打分、无额外 NMS/TTA/置信度过滤，COCO bbox AP50:95/AP75。",
                 "检查点严格加载；每组先与既有源域预测作单图一致性检查。分中心和负样本固定 0.5 阈值诊断在各组子目录。",
                 "","没有发现与 Kvasir 的编码／像素完全重复或 dHash256 距离≤12候选；这不证明患者或视频完全独立。",
                 "本结果属于当前单种子与本地有效标注子集，不能直接当作论文官方分割基准或普遍泛化保证。", "",
                 "来源：[PolypGen 官方仓库](https://github.com/DebeshJha/PolypGen)、[数据论文](https://www.nature.com/articles/s41597-023-01981-y)。"]
        (root/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        state["status"]="completed"
    except BaseException as error:
        state.update(status="failed",error=repr(error));raise
    finally:
        save()


def launch(data, image_root, output):
    project=Path.cwd().resolve();root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(Path(data),root/"data")
    source_files=["polypbench/polypgen.py","polypbench/zeroshot.py","polypbench/adapters/zeroshot_worker.py",
                  "polypbench/evaluation.py","polypbench/data.py","polypbench/io.py"]
    for file in source_files:
        target=root/"source"/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(project/file,target)
    protocol={"created_at":dt.datetime.now(dt.timezone.utc).isoformat(),"project_root":str(project),
              "image_root":str(Path(image_root).resolve()),"runs":RUNS,
              "weight_selection":"preexisting maximum Kvasir validation bbox AP, never PolypGen",
              "training_or_tuning_on_target":False,"inference_images":1537,"primary_scored_images":1473,
              "coordinate_policy":"bundled regionprops generator: zero-based exclusive maximum, no VOC -1/+1 conversion",
              "evaluation_source_sha256":{f:sha256(project/f) for f in source_files},
              "data_sha256":{p.name:sha256(p) for p in (root/"data").glob("*.json")},
              "primary":"all supplied-box still frames, empty text files are negatives",
              "sensitivity":"exclude every copy of internal exact-pixel duplicates",
              "test_parameters":"inherit source-trained config 800/1333, maxDets 100, no extra filtering/NMS/TTA"}
    write_json(root/"protocol.json",protocol)
    env=os.environ.copy();env["PYTHONPATH"]=str(project)+os.pathsep+env.get("PYTHONPATH","")
    env["PYTHONUNBUFFERED"]="1"
    with (root/"pipeline.log").open("w",encoding="utf-8") as log:
        child=subprocess.Popen([sys.executable,"-m","polypbench.zeroshot","run","--output",str(root)],
            cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write_json(root/"launcher.json",{"pid":child.pid,"output":str(root)})
    print(f"Detached zero-shot PID {child.pid}: {root}",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("launch","run"))
    parser.add_argument("--data");parser.add_argument("--image-root");parser.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.action=="launch":launch(args.data,args.image_root,args.output)
    else:run(args.output)


if __name__=="__main__":
    main()
