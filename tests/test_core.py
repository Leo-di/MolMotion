import unittest
import numpy as np
from molmotion.metrics import metrics

class MetricTests(unittest.TestCase):
    def test_identical_trajectories_have_zero_distribution_error(self):
        x=np.random.default_rng(7).normal(size=(2,17,4,14,3))
        values,arrays=metrics(x,x.copy(),np.zeros((2,4),dtype=int),np.ones((2,4,14)))
        for key in ['velocity_sliced_w1','acf_mae','phi_psi_js','velocity_psd_l1','stochastic_coordinate_rmsd']:
            self.assertAlmostEqual(values[key],0.,places=10)
        self.assertEqual(arrays['bond_valid_by_time'].shape,(17,3))

class ModelTests(unittest.TestCase):
    def test_hierarchical_reconstruction_and_seed_reproducibility(self):
        import torch
        from molmotion.model import HierarchicalDynamics, generate
        torch.set_num_threads(1)
        model=HierarchicalDynamics(width=32,slow_layers=1,fast_layers=1,heads=4,checkpointing=False)
        initial=torch.zeros(1,4,21);geometry=torch.zeros(1,4,7)
        sequence=torch.zeros(1,4,dtype=torch.long);mask=torch.ones_like(initial)
        args=(model,initial,geometry,sequence,mask,16,42)
        x,s,r=generate(*args,slow_steps=2,fast_steps=2)
        y,_,_=generate(*args,slow_steps=2,fast_steps=2)
        self.assertEqual(x.shape,(1,17,4,21))
        torch.testing.assert_close(x[:,0],initial)
        torch.testing.assert_close(x,y)
        torch.testing.assert_close(r.reshape(1,2,8,4,21).mean(2),torch.zeros(1,2,4,21),atol=1e-6,rtol=0)
        torch.testing.assert_close(x[:,1:],s.repeat_interleave(8,1)+r*model.residual_scale)
    def test_single_scale_initial_condition(self):
        import torch
        from molmotion.model import SingleScaleDynamics, generate
        m=SingleScaleDynamics(width=32,slow_layers=1,fast_layers=1,heads=4,checkpointing=False);m.joint_ode=2
        initial=torch.zeros(1,4,21)
        x,_,_=generate(m,initial,torch.zeros(1,4,7),torch.zeros(1,4,dtype=torch.long),torch.ones_like(initial),8,42)
        self.assertTrue(torch.isfinite(x).all())
        torch.testing.assert_close(x[:,0],initial)

if __name__=='__main__':unittest.main()
