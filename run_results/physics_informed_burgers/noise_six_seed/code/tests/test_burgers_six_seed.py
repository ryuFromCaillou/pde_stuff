"""Aggregation must reject missing/misaligned evidence and avoid signed-error cancellation."""
import unittest
import numpy as np
import pandas as pd
from utils.burgers_six_seed import average_frames

class AggregationTests(unittest.TestCase):
    def frames(self):
        return [pd.DataFrame({'step':[0,10,20],'transport':[-1.,-1.+(-1)**i*.2,-1.],
                              'transport_abs_error':[0.,.2,0.]}) for i in range(6)]
    def test_signed_means_do_not_replace_mean_errors(self):
        mean,std=average_frames(self.frames())
        self.assertAlmostEqual(mean.transport.iloc[1],-1.)
        self.assertAlmostEqual(mean.transport_abs_error.iloc[1],.2)
        self.assertAlmostEqual(std.transport.iloc[1],.2)
    def test_six_required(self):
        for n in (0,1,5):
            with self.assertRaises(ValueError):average_frames(self.frames()[:n])
    def test_coordinates_must_match(self):
        f=self.frames();f[-1].loc[1,'step']=11
        with self.assertRaises(ValueError):average_frames(f)
    def test_no_truncation(self):
        f=self.frames();f[-1]=f[-1].iloc[:-1]
        with self.assertRaises(ValueError):average_frames(f)
    def test_definitions_must_match(self):
        f=self.frames();f[-1]=f[-1].rename(columns={'transport':'diffusion'})
        with self.assertRaises(ValueError):average_frames(f)
    def test_nonfinite_rejected(self):
        f=self.frames();f[-1].loc[0,'transport']=np.nan
        with self.assertRaises(ValueError):average_frames(f)

if __name__=='__main__':unittest.main()
