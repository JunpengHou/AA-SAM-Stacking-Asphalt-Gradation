import unittest
import numpy as np
import pandas as pd
from gradation.segmentation import instances_from_candidates,evaluate_instances,boundary_f1
from gradation.analysis import field_validation,eta_squared,regression_metrics,cross_section_metrics
from gradation.database import TARGETS

class SegmentationAnalysisTests(unittest.TestCase):
    def test_instances_nms_overlap_empty(self):
        masks=np.zeros((3,8,8),bool);masks[0,1:4,1:4]=1;masks[1]=masks[0];masks[2,3:6,3:6]=1
        labels,rows=instances_from_candidates(masks,[.9,.8,.7])
        self.assertEqual(len(rows),2);self.assertEqual(labels.max(),2)
        self.assertEqual(rows[1]["overlap_removed_px"],1)
        labels,rows=instances_from_candidates(np.zeros((0,8,8)),[])
        self.assertFalse(rows);self.assertEqual(labels.sum(),0)

    def test_ap_ranking_and_boundary(self):
        gt=np.zeros((20,20),int);gt[2:10,2:10]=1
        false=np.zeros_like(gt,bool);false[12:19,12:19]=1
        masks=np.array([false,gt>0])
        sample=dict(image_id="x",truth=gt,masks=masks,scores=[.9,.8])
        result=evaluate_instances([sample])
        self.assertAlmostEqual(result["Mask_AP50_all_points"],.5)
        self.assertAlmostEqual(result["F1"],2/3)
        sample["scores"]=[.8,.9]
        self.assertAlmostEqual(evaluate_instances([sample])["Mask_AP50_all_points"],1.)
        shifted=np.roll(gt>0,1,axis=0)
        self.assertLess(boundary_f1(shifted,gt>0,0),1)
        self.assertEqual(boundary_f1(shifted,gt>0,2),1.)
        self.assertEqual(evaluate_instances([sample],score_threshold=.95)["TP"],0)
        self.assertAlmostEqual(evaluate_instances([sample],score_threshold=.95)["Mask_AP50_all_points"],1.)

    def test_field_hierarchy(self):
        patches=[];cores=[]
        for road in range(3):
            for point in range(4):
                ids=dict(road_id=f"r{road}",cross_section_id="c",point_id=str(point))
                cores.append({**ids,**dict.fromkeys(TARGETS,10.)})
                for patch in range(4):patches.append({**ids,"patch_id":str(patch),**dict.fromkeys(TARGETS,8.+patch)})
        p,s,m=field_validation(pd.DataFrame(patches),pd.DataFrame(cores))
        self.assertEqual(len(p),12);self.assertEqual(len(s),3)
        self.assertAlmostEqual(m.MSE.iloc[0],.25)
        section_metrics=cross_section_metrics(s)
        self.assertEqual(len(section_metrics),3)
        self.assertTrue(section_metrics.n_sieves.eq(6).all())
        self.assertTrue(np.allclose(section_metrics.MAPE,5.))
        with self.assertRaises(ValueError):field_validation(pd.DataFrame(patches[:-1]),pd.DataFrame(cores))
        self.assertIsNone(regression_metrics([0,1],[1,1])["MAPE"])

    def test_eta_squared(self):
        table=pd.DataFrame(dict(gradation_id=["a","a","b","b"],x=[1.,1.,3.,3.],c=[2.,2.,2.,2.]))
        out=eta_squared(table,["x","c"])
        self.assertAlmostEqual(out.eta_squared.iloc[0],1.)
        self.assertTrue(np.isnan(out.eta_squared.iloc[1]))

if __name__=="__main__":unittest.main()
