import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from detectron2.engine.hooks import BestCheckpointer
from methods.quality_alignment.resume import restore_trainer, rollback_logs


class ResumeTests(unittest.TestCase):
    def test_restore_counter_best_selection_and_native_data_cursor(self):
        model=torch.nn.Module();model.class_embed=torch.nn.Linear(2,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
        for parameter in model.parameters():optimizer.state[parameter]['step']=torch.tensor(539.)
        hook=BestCheckpointer(275,None,'bbox/AP',mode='max')
        visits=[]
        def batches():
            for index in range(8801):visits.append(index);yield index
        trainer=SimpleNamespace(iter=0,model=model,optimizer=optimizer,_hooks=[hook],
            scheduler=SimpleNamespace(state_dict=lambda:{'last_epoch':550}),
            _trainer=SimpleNamespace(accumulation_steps=16,grad_scaler=torch.amp.GradScaler('cpu',init_scale=32),
                _data_loader_iter=iter(batches()),successful_updates=0))
        trainer.checkpointer=SimpleNamespace(load=lambda _:setattr(trainer,'iter',549))
        best={'iteration':549,'metric_percent':50.00875479}
        with tempfile.TemporaryDirectory() as tmp:
            artifact=Path(tmp);checkpoint=artifact/'checkpoint.pth';checkpoint.write_bytes(b'fixture')
            restore_trainer(trainer,checkpoint,best,artifact)
            report=json.loads((artifact/'restored.json').read_text())
        self.assertEqual(trainer.start_iter,550);self.assertEqual(trainer._trainer.successful_updates,539)
        self.assertEqual(hook.best_iter,549);self.assertEqual(hook.best_metric,50.00875479)
        self.assertEqual(len(visits),8800);self.assertEqual(next(trainer._trainer._data_loader_iter),8800)
        self.assertTrue(report['parameter_gradients_empty']);self.assertEqual(report['grad_scaler']['scale'],32)

    def test_rollback_removes_only_uncheckpointed_training_and_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);artifact=root/'artifact';artifact.mkdir()
            metrics=[{'iteration':539,'optimizer/successful_updates':529},
                     {'iteration':550,'bbox/AP':50.0},{'iteration':559,'optimizer/successful_updates':549}]
            probes=[{'iteration':275},{'iteration':550}]
            for filename,rows in (('metrics.json',metrics),('quality_gradient_probes.jsonl',probes)):
                (root/filename).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            rollback_logs(root,artifact,550)
            actual=[json.loads(line) for line in (root/'metrics.json').read_text().splitlines()]
            self.assertEqual(actual,metrics[:2])
            self.assertEqual(json.loads((root/'quality_gradient_probes.jsonl').read_text()),probes[0])


if __name__=='__main__':unittest.main()
