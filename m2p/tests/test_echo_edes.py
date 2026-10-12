"""Checks that protect the new ED/ES data labels and coordinate association."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_echoxflow_edes import align_phases, select_phases


def contour_cycle():
    # The largest peak occurs AFTER ES; the chosen ED must precede ES.
    radius = np.array([1., 1., .94, .83, .7, .6, .6, .7, .85, 1.04, 1.05])
    angle = np.linspace(0, 2*np.pi, 30, endpoint=False)
    boundary = np.column_stack((np.cos(angle), np.sin(angle)))
    layers = boundary[:, None, :] * np.linspace(1., 1.2, 5)[None, :, None]
    return (radius[:, None, None, None] * layers[None]).reshape(len(radius), -1, 2)


class Phases(unittest.TestCase):
    def test_choose_preceding_diastole_not_next_cycle_peak(self):
        times = np.linspace(0, 1, 11)
        selected, details = select_phases(contour_cycle(), times, [0., 1.])
        self.assertEqual(selected[0], 0)
        self.assertIn(selected[1], [5, 6])
        self.assertLess(selected[0], selected[1])
        self.assertEqual(details["label_status"], "estimated_not_official")

    def test_flat_area_cannot_be_labelled_ED_ES(self):
        contours = np.repeat(contour_cycle()[:1], 11, axis=0)
        with self.assertRaises(ValueError):
            select_phases(contours, np.linspace(0,1,11), [0.,1.])

    def test_alignment_rejects_different_time_origin(self):
        with self.assertRaises(ValueError):
            align_phases(np.array([0., .5]), np.linspace(10,11,51), np.array([0,1]))

    def test_alignment_preserves_frame_indices(self):
        bt=np.linspace(12,13,51)
        ct=bt[[10,30]]+.003
        indices, errors, tolerance=align_phases(ct,bt,np.array([0,1]))
        np.testing.assert_array_equal(indices,[10,30])
        self.assertTrue((errors<tolerance).all())

    def test_alignment_rejects_same_target_frame(self):
        with self.assertRaises(ValueError):
            align_phases(np.array([0., .001]), np.linspace(0,1,51), np.array([0,1]))


if __name__=="__main__":
    unittest.main()
