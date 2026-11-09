import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class ModelComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0,str(ROOT/"AA-SAM"))
        from aggregate_sam3 import model
        cls.module=model

    def test_afeb_shapes_and_fusion_initialization_backward(self):
        import torch
        m=self.module
        torch.manual_seed(5)
        branch=m.AggregateFeatureBranch()
        outputs=branch(torch.rand(2,3,64,64))
        self.assertEqual([tuple(x.shape) for x in outputs],[(2,256,16,16),(2,256,8,8),(2,256,4,4),(2,256,2,2)])
        for layer in branch.modules():
            if isinstance(layer,m.DepthwiseResidualBlock):self.assertTrue(torch.allclose(layer.layer_scale,torch.full_like(layer.layer_scale,1e-4)))
        fusion=m.GatedResidualFusion()
        original=torch.randn(2,256,8,8)
        fused=fusion(original,outputs[0])
        self.assertTrue(torch.equal(fused,original))
        self.assertAlmostEqual(float(fusion.scale.detach()),.05,places=6)
        loss=fused.square().mean();loss.backward()
        self.assertGreater(float(fusion.residual[-1].weight.grad.abs().sum()),0.)
        optimizer=torch.optim.AdamW(list(branch.parameters())+list(fusion.parameters()),lr=1e-3)
        optimizer.step();optimizer.zero_grad()
        features=branch(torch.rand(2,3,64,64))
        fusion(original,features[0]).square().mean().backward()
        self.assertGreater(float(branch.stem[0][0].weight.grad.abs().sum()),0.)

    def test_adapter_parameter_freezing(self):
        import torch
        from torch import nn
        model=nn.Module();model.backbone=nn.Module();model.backbone.language_backbone=nn.Linear(2,2)
        model.backbone.vision_backbone=nn.Linear(2,2)
        model.aggregate_branch=nn.Linear(2,2);model.fusion_blocks=nn.Linear(2,2)
        model.transformer=nn.Module();model.transformer.decoder=nn.Linear(2,2);model.transformer.encoder=nn.Linear(2,2)
        model.segmentation_head=nn.Linear(2,2)
        self.module._set_freeze_mode(model,"adapter")
        for name,param in model.named_parameters():
            target=any(key in name for key in ["aggregate_branch","fusion_blocks","transformer.decoder","segmentation_head"])
            self.assertEqual(param.requires_grad,target,name)

    def test_legacy_evaluation_rejects_test_sweep_and_uses_euclidean_tolerance(self):
        for name in ["evaluate_all_checkpoints.py","evaluate_all_checkpoints_mud_fixed.py"]:
            spec=importlib.util.spec_from_file_location("_legacy_eval",ROOT/"AA-SAM"/"scripts"/name)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            args=[name,"--checkpoint-dir","missing","--dataset-root","missing","--sam3-root","missing","--split","test"]
            with patch.object(sys,"argv",args),patch.object(module,"ensure_dependencies"):
                with self.assertRaisesRegex(ValueError,"validation-only"):module.main()
            # Diagonal distance sqrt(8)>2 must NOT match a 2-pixel tolerance.
            pred=np.zeros((10,10),bool);truth=pred.copy();pred[3,3]=1;truth[5,5]=1
            self.assertEqual(module.boundary_f1(pred,truth,2),0.)

if __name__=="__main__":unittest.main()
