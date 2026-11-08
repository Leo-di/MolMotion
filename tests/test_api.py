import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from molmotion import evaluate_npz

class EvaluationAPITests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.path=Path(self.directory.name)/'input.npz'
        x=np.random.default_rng(11).normal(size=(2,17,4,14,3))
        self.data=dict(generated=x,reference=x.copy(),sequence_ids=np.zeros((2,4),dtype=int),atom_mask=np.ones((2,4,14)))
    def tearDown(self):self.directory.cleanup()
    def test_cli_matches_python_interface(self):
        np.savez(self.path,**self.data)
        p=subprocess.run([sys.executable,'-m','molmotion.cli','evaluate',str(self.path)],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(p.stdout),evaluate_npz(self.path))
    def test_rejects_missing_coordinates(self):
        np.savez(self.path,bond_valid_by_time=np.ones(17))
        with self.assertRaisesRegex(ValueError,'Missing fields'):evaluate_npz(self.path)
    def test_rejects_nonfinite_coordinates(self):
        self.data['generated'][0,0,0,0,0]=np.nan
        np.savez(self.path,**self.data)
        with self.assertRaisesRegex(ValueError,'finite'):evaluate_npz(self.path)
    def test_rejects_missing_backbone_atoms(self):
        self.data['atom_mask'][0,0,0]=0
        np.savez(self.path,**self.data)
        with self.assertRaisesRegex(ValueError,'backbone'):evaluate_npz(self.path)

    def test_example_is_available_from_installed_package(self):
        output=Path(self.directory.name)/"example.pdb"
        subprocess.run([sys.executable,"-m","molmotion.cli","example","--output",str(output)],check=True,capture_output=True)
        self.assertIn("ATOM",output.read_text())
