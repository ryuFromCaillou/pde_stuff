"""Physical units, coupled gradients, clean data, and recovery boundary regressions."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from utils.physics_informed_burgers import (PhysicalHead, coefficients, dataset, initialize, validate)


class PhysicsInformedBurgersTests(unittest.TestCase):
    def test_mechanical_and_numerical_validation(self):
        self.assertTrue(validate()['passed'])

    def test_physical_head_and_strict_thresholds(self):
        h=PhysicalHead()
        self.assertEqual(sum(p.numel() for p in h.parameters()),2)
        self.assertEqual(h.xi.detach().tolist(),[0.,0.])
        with torch.no_grad(): h.xi.copy_(torch.tensor([-1.,.02]))
        f=torch.tensor([[2.,3.,4.]])
        torch.testing.assert_close(h(f),torch.tensor([[-5.92]]))
        self.assertTrue(coefficients(h)['strong'])
        with torch.no_grad(): h.xi[0]=-.75
        self.assertFalse(coefficients(h)['loose'])

    def test_fixed_clean_data_and_independent_initialization(self):
        train,full,_,grid=dataset()
        self.assertEqual(len(train[0]),640)
        self.assertEqual(grid['u'].shape,(51,256))
        for a,b in zip(train,full): torch.testing.assert_close(a,b[grid['indices']],rtol=0,atol=0)
        a,_=initialize(0); b,_=initialize(0); c,_=initialize(1)
        for p,q in zip(a.parameters(),b.parameters()): self.assertTrue(torch.equal(p,q))
        self.assertFalse(torch.equal(next(a.parameters()),next(c.parameters())))
        np.testing.assert_array_equal(grid['indices'],np.unique(grid['indices']))

    def test_single_repeat_summary_uses_null_sample_std(self):
        import pandas as pd
        from utils.physics_informed_burgers import summarize
        fields=('transport','diffusion','observation_mse','u_mse','u_rel_l2',
                'ut_rel_l2','ux_rel_l2','uxx_rel_l2','independent_pde_mse',
                'observation_total_loss','data_loss','pde_loss')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); summaries=[]
            for seed in (0,1):
                row={k:.1 for k in fields}
                row.update(seed=seed,step=0,first_loose_step=None,first_strong_step=None,
                           status='not_recovered',loose=False,strong=False)
                summaries.append(row)
                folder=root/f'seed_{seed}'; folder.mkdir()
                pd.DataFrame([row]).to_csv(folder/'history.csv',index=False)
                (folder/'checkpoint_metrics.json').write_text(json.dumps([row]))
            summarize(root,summaries)
            result=json.loads((root/'seed_distribution.json').read_text())
            self.assertIsNone(result['distributions']['transport']['std'])
            self.assertEqual(result['repeats'],1)


if __name__=='__main__': unittest.main()
