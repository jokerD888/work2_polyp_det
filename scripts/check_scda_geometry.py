"""Explicit GPU diagnostic; invoke only after baseline has released the GPU."""

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='methods/scda_proposal/configs/kvasir.yaml')
    parser.add_argument('--output', default='runs/scda-real-geometry.json')
    args = parser.parse_args()
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party'/'MaskDINO'))
    import copy,sys,torch,numpy as np,random,json,pathlib
    from polypbench.io import write_json
    from methods.scda_proposal.experiment import load_experiment
    from methods.scda_proposal.precheck import build_cfg
    from methods.scda_proposal.integration import register
    from methods.scda_proposal.mapper import SCDALSJMapper
    from methods.scda_proposal.scda_assigner import token_geometry, fully_valid_cells
    from methods.scda_proposal.scda_loss import cxcywh_to_xyxy
    from detectron2.data import DatasetCatalog,MetadataCatalog,detection_utils
    from detectron2.data.datasets import load_coco_json
    from detectron2.modeling import build_model
    from detectron2.structures import Instances,Boxes
    from detectron2.utils.events import EventStorage
    if Path(args.output).exists(): raise FileExistsError(args.output)
    cfg=load_experiment(args.config)
    sys.path.insert(0,cfg['repo']['path'])
    import train_net
    register(cfg['repo']['path'])
    e=cfg['dataset']['train']
    DatasetCatalog.register('bench_train',lambda:load_coco_json(e['annotations'],e['image_root'],'bench_train',extra_annotation_keys=['id']))
    MetadataCatalog.get('bench_train').set(json_file=e['annotations'],image_root=e['image_root'],evaluator_type='coco')
    c=build_cfg(cfg,True,pathlib.Path('runs/scda-real-geometry'))
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    model=build_model(c).cuda().train()
    mapper=SCDALSJMapper(c,'single_instance')
    source=[r for r in DatasetCatalog.get('bench_train') if len(r['annotations'])==1][:2]
    batch=[]
    for raw,(h,w) in zip(source,[(319,447),(383,511)]):
        record=mapper(raw); original=record['instances']
        record['image']=record['image'][:,:h,:w].contiguous()
        record['padding_mask']=record['padding_mask'][:h,:w].contiguous()
        inst=Instances((h,w))
        inst.gt_boxes=Boxes(original.gt_boxes.tensor.clone());inst.gt_boxes.clip((h,w))
        for name,value in original.get_fields().items():
            if name=='gt_boxes':continue
            inst.set(name,value[:,:h,:w].clone() if name=='gt_masks' else value.clone())
        record['instances']=detection_utils.filter_empty_instances(inst,by_mask=False)
        batch.append(record)
    observations=[]
    def callback(m,native,aux,payload,targets,assignments):
        for b,(target,assignment) in enumerate(zip(targets,assignments)):
            cells,scale,keys=token_geometry(payload,b,target)
            h,w=target.image_size
            anchor=payload['base_anchor'][b,:,:2]*scale[:,:2]*torch.tensor([w,h],device='cuda')
            valid=payload['valid_token_mask'][b]
            torch.testing.assert_close(anchor[valid].double(),((cells[:,:2]+cells[:,2:])/2)[valid],rtol=1e-6,atol=5e-5)
            truth=cxcywh_to_xyxy(target.boxes)*torch.tensor([w,h,w,h],device='cuda')
            torch.testing.assert_close(truth,batch[b]['instances'].gt_boxes.tensor.cuda(),rtol=1e-6,atol=1e-4)
            for chosen in (assignment.cls.token_indices,assignment.reg.token_indices):
                assert fully_valid_cells(target.valid_pixels,cells[chosen]).all()
            observations.append({'image_size':[h,w],'padded_size':list(target.valid_pixels.shape),'spatial_shapes':payload['spatial_shapes'].tolist(),'cls':assignment.cls.token_indices.numel(),'reg':assignment.reg.token_indices.numel()})
    model.scda_audit_callback=callback
    with EventStorage():
        with torch.autocast('cuda',enabled=True):losses=model(batch)
        total=sum(losses.values());assert torch.isfinite(total);total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    write_json(args.output,{'passed':True,'scda_options':cfg['scda'],'real_variable_batch':observations,'note':'Controlled pixel cropping, native ImageList padding, actual GT/anchor/cell coordinates, AMP backward; no source dataset mutation'})
    print(json.dumps(observations))


if __name__ == '__main__':
    main()
