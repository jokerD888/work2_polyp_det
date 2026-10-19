import copy
import unittest
import torch
from torch import nn
from detectron2.utils.events import EventStorage
from methods.mssl.accumulation import AccumulatingAMPTrainer


@unittest.skipUnless(torch.cuda.is_available(),"CUDA required")
class AccumulationTests(unittest.TestCase):
    def test_averaging_and_one_optimizer_step(self):
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=nn.Parameter(torch.tensor(.1,device="cuda"))
                self.criterion=nn.Identity()
            def forward(self,data):
                return {"loss":(self.weight*data["x"]-data["y"]).square().mean()}
        model=SmallModel();control=copy.deepcopy(model)
        batches=[{"x":torch.tensor([i/10],device="cuda"),
                  "y":torch.tensor([i/20],device="cuda")} for i in range(1,5)]
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        trainer=AccumulatingAMPTrainer(model,batches,optimizer,4)
        with EventStorage():
            trainer.run_step()
        expected=sum(control(x)["loss"] for x in batches)/4
        expected.backward()
        with torch.no_grad():control.weight-=.01*control.weight.grad
        torch.testing.assert_close(model.weight,control.weight,rtol=1e-6,atol=1e-7)
        self.assertEqual(trainer.successful_updates,1)


if __name__=="__main__":
    unittest.main()
