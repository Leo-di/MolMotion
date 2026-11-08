import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from molmotion import MolMotionPipeline
from molmotion.model import HierarchicalDynamics
from molmotion.pipeline import _sha,load_pdb,encode_initial

PDB=Path(__file__).resolve().parents[1]/'examples/IMCP.pdb'

class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        model=HierarchicalDynamics(width=32,slow_layers=1,fast_layers=1,heads=4,checkpointing=False)
        save_file(model.state_dict(),str(self.root/'model.safetensors'))
        config=dict(format_version=1,feature_version='mdgen-offset21-v1',stride=8,architecture='hierarchical',width=32,slow_layers=1,fast_layers=1,heads=4,frame_interval_ps=10.,future_frames=16,weights='model.safetensors',weights_sha256=_sha(self.root/'model.safetensors'))
        (self.root/'config.json').write_text(json.dumps(config))
    def tearDown(self):self.tmp.cleanup()
    def test_model_pdb_and_xtc_roundtrip(self):
        import mdtraj as md
        pipe=MolMotionPipeline.from_pretrained(self.root)
        result=pipe.generate(PDB,num_frames=16,num_samples=2,slow_steps=2,fast_steps=2)
        x,_,_=load_pdb(PDB)
        np.testing.assert_allclose(result.coordinates[:,0],np.repeat(x[None],2,axis=0))
        self.assertEqual(result.coordinates.shape,(2,17,4,14,3))
        result.save_npz(self.root/'out.npz');result.save_pdb(self.root/'out.pdb');result.save_xtc(self.root/'out.xtc')
        p=md.load(str(self.root/'out.pdb'));t=md.load(str(self.root/'out.xtc'),top=str(self.root/'out.pdb'))
        self.assertEqual(p.n_atoms,29);self.assertEqual(t.n_frames,17)
        np.testing.assert_allclose(t.time,np.arange(17)*10.)
        # XTC defaults to 0.001 nm quantization; PDB rounds to 0.001 Angstrom.
        np.testing.assert_allclose(p.xyz,t.xyz,atol=.00056,rtol=0)
        self.assertLess(result.evaluate(result.coordinates.copy())['velocity_sliced_w1'],1e-10)
    def test_bad_weight_hash_is_rejected(self):
        with (self.root/'model.safetensors').open('ab') as f:f.write(b'x')
        with self.assertRaisesRegex(ValueError,'SHA256'):MolMotionPipeline.from_pretrained(self.root)
    def test_initial_encoding_matches_training_geometry(self):
        from molmotion._vendor.mdgen.rigid_utils import Rigid
        from molmotion._vendor.mdgen.geometry import frames_torsions_to_atom14
        x,s,m=load_pdb(PDB);raw,geom,seq,mask,frames=encode_initial(x,s,'cpu')
        offsets=raw[...,:7]
        decoded=frames.compose(Rigid.from_tensor_7(offsets,normalize_quats=True))
        tor=raw[...,7:].reshape(1,4,7,2)
        direct=frames_torsions_to_atom14(frames,tor,seq)
        indirect=frames_torsions_to_atom14(decoded,tor,seq)
        torch.testing.assert_close(direct,indirect,atol=1e-5,rtol=1e-5)
        self.assertTrue(torch.isfinite(geom).all())
    def test_unvalidated_horizon_is_rejected(self):
        pipe=MolMotionPipeline.from_pretrained(self.root)
        with self.assertRaisesRegex(ValueError,'horizon'):pipe.generate(PDB,num_frames=256)
