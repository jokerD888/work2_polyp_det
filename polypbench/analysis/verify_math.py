"""Numerically align the frozen trained V1 implementation with the derivation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from polypbench.io import write_json,sha256


def run(workspace,output):
    torch.set_num_threads(2)
    source=Path(workspace)/"runs/mssl-implementation-20261018/campaign-r2/source/methods/mssl/loss.py"
    spec=importlib.util.spec_from_file_location("actual_trained_v1_loss",source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    options=SimpleNamespace(support="mask",temperature=.05,direction_bandwidth=.15,min_scale_pixels=4.)
    yy,xx=np.mgrid[:64,:64]; mask=(((xx-31)/19)**2+((yy-30)/23)**2<=1).astype(np.float32)
    checks=[]
    for target in (np.array([12.,7.,51.,54.]),np.array([5.,2.,59.,61.])):
        size=np.maximum(target[2:]-target[:2],4); scale=np.r_[size,size]
        for d in (-.5,-.01,0.,.01,.5):
            pred=torch.tensor(target+d*scale,dtype=torch.float32,requires_grad=True)
            actual=module.side_losses(pred,torch.tensor(target,dtype=torch.float32),torch.tensor(mask),options)
            grad=torch.autograd.grad(actual.sum(),pred)[0].numpy()*scale
            expected=[]; expected_grad=[]
            for e,axis in enumerate((0,1,0,1)):
                mass=mask.sum(axis=0 if axis==0 else 1).astype(float); good=mass>0
                u=np.arange(len(mass))[good]+.5; r=(u-target[e])/size[axis]
                logpi=np.log(mass[good])-abs(r)/.15; pi=np.exp(logpi-logpi.max()); pi/=pi.sum()
                sigmoid=lambda z:1/(1+np.exp(np.clip(-z,-700,700)))
                v=sigmoid((d-r)/.05); z=sigmoid(-r/.05)
                diff=float(pi@(v-z)); slope=float(pi@(v*(1-v))/.05)
                expected.append(.5*diff**2); expected_grad.append(diff*slope)
            np.testing.assert_allclose(actual.detach().numpy(),expected,rtol=5e-4,atol=2e-7)
            np.testing.assert_allclose(grad,expected_grad,rtol=5e-4,atol=2e-6)
            if d==0: assert np.all(actual.detach().numpy()==0) and np.all(grad==0)
            else: assert np.all(np.asarray(expected_grad)*d>0)
            checks.append({"target":target.tolist(),"d":d,"max_loss_error":float(max(abs(actual.detach().numpy()-expected))),
                           "max_gradient_error":float(max(abs(grad-expected_grad)))})
    write_json(output,{"status":"passed","actual_trained_loss_sha256":sha256(source),
        "checks":checks,"scope":"CPU FP32 trained code vs float64 analytic G and dL/dd; provided-box/mask margin included; elementary properties do not guarantee network convergence or AP"})
