import unittest
import os

import numpy as np

import sdt.brightness
import sdt.background
import sdt.sim


path, f = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(path, "data_background")


class TestWavelet(unittest.TestCase):
    def setUp(self):
        pos = np.array([[10, 10], [20, 20], [30, 30], [40, 40]])
        amp = 200
        sigma = 1
        width = 60
        height = 50
        self.img = sdt.sim.simulate_gauss((width, height), pos, amp, sigma)
        m = np.meshgrid(np.arange(-width//2, width//2),
                        np.arange(-height//2, height//2))
        self.bg = 0.005*m[0]**3 + 0.005*m[1]**3 - 0.01*m[0]**2 - 0.01*m[1]**2

        self.wavelet_options = dict(threshold=5, wtype="db4", wlevel=2,
                                    ext_mode="smooth", max_iterations=20,
                                    detail=0, conv_threshold=5e-3)

        # created from a test run
        self.orig = np.load(os.path.join(data_path, "wavelet_bg.npz"))
        self.orig = self.orig["bg_est"]

    def test_estimate_bg(self):
        bg_est = sdt.background.estimate_bg_wavelet(
            self.bg+self.img, **self.wavelet_options)
        np.testing.assert_allclose(bg_est, self.orig, atol=1e-3)

    def test_remove_bg(self):
        img_est = sdt.background.remove_bg_wavelet(
            self.bg+self.img, **self.wavelet_options)
        np.testing.assert_allclose(img_est, self.img+self.bg-self.orig)


if __name__ == "__main__":
    unittest.main()
