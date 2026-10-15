"""Explicit GPU diagnostic; invoke only after baseline has released the GPU."""

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='methods/scda_proposal/configs/kvasir.yaml')
    parser.add_argument('--output', default='runs/SCDA_DECOUPLING_MATURE_BASELINE.json')
    parser.add_argument('--checkpoint', default='runs/maskdino-official-s42/train/model_best.pth')
    args = parser.parse_args()
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party'/'MaskDINO'))
    import os,sys,json,pathlib,torch,numpy as np,random
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    from polypbench.io import write_json,sha256
    from methods.scda_proposal.experiment import load_experiment
    from methods.scda_proposal.precheck import build_cfg
    from methods.scda_proposal.integration import register
    from methods.scda_proposal.mapper import SCDALSJMapper
    from methods.scda_proposal.audit_summary import summarize_decoupling, validate_v2_assignments
    from methods.scda_proposal.diagnostics import proposal_diagnostics
    from detectron2.data import DatasetCatalog,MetadataCatalog
    from detectron2.data.datasets import load_coco_json
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.utils.events import EventStorage
    if Path(args.output).exists(): raise FileExistsError(args.output)
    cfg=load_experiment(args.config)
    sys.path.insert(0,cfg['repo']['path'])
    import train_net
    register(cfg['repo']['path'])
    e=cfg['dataset']['train']
    DatasetCatalog.register('bench_train',lambda:load_coco_json(e['annotations'],e['image_root'],'bench_train',extra_annotation_keys=['id']))
    MetadataCatalog.get('bench_train').set(json_file=e['annotations'],image_root=e['image_root'],evaluator_type='coco')
    c=build_cfg(cfg,True,Path(args.output).parent)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    model=build_model(c)
    checkpoint=args.checkpoint
    DetectionCheckpointer(model).load(checkpoint)
    model.cuda().train()
    records=DatasetCatalog.get('bench_train')
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    order=torch.randperm(len(records),generator=torch.Generator().manual_seed(42)).tolist()[:128]
    mapper=SCDALSJMapper(c,'single_instance')
    mapped=[mapper(records[index]) for index in order]
    images=[];proposals=[]
    def callback(m,native,aux,payload,targets,assignments):
        images.extend(assigned.audit for assigned in assignments)
        proposals.extend(proposal_diagnostics(payload,targets,assignments))
    model.scda_audit_callback=callback
    with EventStorage() as storage:
        with torch.no_grad():
            for start in range(0,len(mapped),4):
                with torch.autocast('cuda',enabled=True):model(mapped[start:start+4])
                storage.step()
    summary=summarize_decoupling(images)
    if cfg['scda']['reg_selection'] == 'core_plus_transition':
        summary['confirmed_v2_policy'] = validate_v2_assignments(images)
    summary.update(parameter_stage='Mature original baseline best checkpoint, offline diagnostic only; NOT F training initialization',checkpoint_sha256=sha256(checkpoint),seed=42,proposal_diagnostics=proposals)
    write_json(args.output,summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('per_image','proposal_diagnostics')}))


if __name__ == '__main__':
    main()
