# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from sdt import io, loc, roi, sim
from sdt.loc.cg import algorithm, find


data_path = Path(__file__).absolute().parent / "data_loc"


@pytest.fixture
def regression_img():
    # Images for regression testing
    with np.load(data_path / "pMHC_AF647_200k_000_.npz") as fr:
        return fr["frames"]


@pytest.fixture
def regression_data():
    # Data produced by original implementation for regression testing
    with np.load(data_path / "cg_loc_orig.npz") as orig:
        ret = []
        for k, v in orig.items():
            r = pd.DataFrame(v, columns=algorithm.peak_params)
            r["frame"] = int(k)
            ret.append(r)
    ret = sorted(ret, key=lambda x: x.loc[0, "frame"])
    return pd.concat(ret, ignore_index=True)


def check_regression_result(res, exp, img_shape):
    frame_col = None

    if isinstance(res, pd.DataFrame):
        for pp in algorithm.peak_params:
            assert pp in res
        if "frame" in res:
            frame_col = res["frame"]
        res = res[algorithm.peak_params].to_numpy()

    radius = regression_options["radius"]
    # The original implementation's `local_maxima` removed maxima that
    # were ``radius + 1`` pixels away from right/bottom edge, which is
    # not necessary; removing maxima ``radius`` or less pixels away
    # from  edges is sufficient.
    yx = res[:, [algorithm.col_nums.y, algorithm.col_nums.x]]
    yx_max_wrong = np.array(img_shape) - radius - 1
    print(yx_max_wrong)
    wrongly_removed = np.any(
        (yx >= yx_max_wrong - 1) & (yx <= yx_max_wrong + 1),
        axis=1,
    )
    res = res[~wrongly_removed]

    if frame_col is not None:
        np.testing.assert_array_equal(frame_col[~wrongly_removed],
                                      exp["frame"])

    # Since subpixel image shifting was fixed, tolerances need to
    # be higher.
    np.testing.assert_allclose(
        res[:, [algorithm.col_nums.x, algorithm.col_nums.y]],
        exp[["x", "y"]], atol=0.05)
    np.testing.assert_allclose(
        res[:, algorithm.col_nums.mass], exp["mass"], rtol=0.02)
    np.testing.assert_allclose(
        res[:, algorithm.col_nums.size]**2, exp["size"], rtol=0.04)
    np.testing.assert_allclose(
        res[:, algorithm.col_nums.ecc], exp["ecc"], atol=0.01)


regression_options = {"radius": 3, "signal_thresh": 300, "mass_thresh": 5000,
                      "bandpass": True}


class TestAlgorithm:
    def test_locate(self):
        """Test with synthetic data"""
        d = np.array([[3.45, 10.0, 1000.0, 0.8],  # close to edge
                      [10.0, 3.48, 1500.0, 0.9],  # close to edge
                      [96.0, 50.0, 1200.0, 1.0],  # close to edge
                      [52.0, 77.5, 1750.0, 1.1],  # close to edge
                      [27.3, 56.7, 1450.0, 1.2],  # good
                      [34.2, 61.4, 950.0, 1.05],  # too dim
                      [62.2, 11.4, 1320.0, 1.05]])  # good
        img = sim.simulate_gauss((100, 80), d[:, [0, 1]], d[:, 2], d[:, 3],
                                 mass=True)
        res = algorithm.locate(img, 4, 10, 1000, bandpass=False)
        res = res[np.argsort(res[:, algorithm.col_nums.x])]

        exp = d[[4, 6], :]
        exp = exp[np.argsort(exp[:, 0])]

        np.testing.assert_allclose(
            res[:, [algorithm.col_nums.x, algorithm.col_nums.y]],
            exp[:, [0, 1]], atol=0.005)
        np.testing.assert_allclose(
            res[:, algorithm.col_nums.mass], exp[:, 2], rtol=0.005)
        # Skip "size" column, this is too unpredictable
        np.testing.assert_allclose(res[:, algorithm.col_nums.ecc], 0.0,
                                   atol=0.03)

    def test_large_shift(self):
        """Make sure a large shift (> 1 pixel) between local maximum and center
        of mass is handled correctly.
        """
        img = np.zeros((80, 100), dtype=int)
        img[10, 12] = 200
        img[10, 13] = 150
        img[11, 14] = 150
        img[11, 15] = 150

        img[11, 85] = 150
        img[11, 86] = 150
        img[10, 87] = 150
        img[10, 88] = 200

        img[69, 85] = 150
        img[69, 86] = 150
        img[70, 87] = 150
        img[70, 88] = 200

        img[70, 12] = 200
        img[70, 13] = 150
        img[69, 14] = 150
        img[69, 15] = 150

        img[3, 50] = 200  # too close to the edge
        img[76, 50] = 200  # too close to the edge
        img[40, 2] = 150
        img[40, 3] = 150
        img[40, 4] = 200  # local max is in bounds, but shifted image is not
        img[40, 95] = 200  # local max is in bounds, but shifted image is not
        img[40, 96] = 150
        img[40, 97] = 150

        res = algorithm.locate(img, 4, 180, 100, bandpass=False)

        # determined by a modified version of the previous algorithm which
        # ignored NaNs coming from shifts > 1 pixel (which is fine if
        # background is zero and radius is large enough)
        x = 13.384615384615383
        y = 10.461538461538462
        mass = 650.0
        rg = 1.4879529376876057
        ecc = 0.567533779144851
        expected = np.array([[x, y, mass, rg, ecc],
                             [100 - x, y, mass, rg, ecc],
                             [x, 80 - y, mass, rg, ecc],
                             [100 - x, 80 - y, mass, rg, ecc]])

        np.testing.assert_allclose(res, expected)

    def test_locate_regression(self, regression_data, regression_img):
        """Compare to result of the original implementation (regression test)
        """
        for i, img in enumerate(regression_img):
            exp = regression_data[regression_data["frame"] == i].copy()
            loc = algorithm.locate(img, **regression_options)
            check_regression_result(loc, exp, img.shape)


class TestAPI:
    @pytest.fixture
    def loc_data(self):
        rs = np.random.RandomState(10)
        d = pd.DataFrame({"x": rs.uniform(10.0, 140.0, 15),
                          "y": rs.uniform(10.0, 90.0, 15),
                          "mass": rs.normal(2000.0, 30.0, 15),
                          "frame": [1] * 9 + [2] * 6,
                          "ecc": 0.0})
        return d

    @contextlib.contextmanager
    def _make_image_sequence(self, ret_type, tmp_path, loc_data):
        if ret_type == "pims":
            pims = pytest.importorskip("pims", reason="pims not installed")
        elif ret_type == "ImageSequence":
            # ImageSequence uses imageio
            pytest.importorskip("imageio", reason="imageio not installed")
        if ret_type in ("pims", "ImageSequence"):
            # Need to write the image
            tifffile = pytest.importorskip(
                "tifffile", reason="tifffile not installed")

        ims = [np.zeros((100, 150), dtype=float)
               for _ in range(loc_data["frame"].max() + 1)]
        for f in loc_data["frame"].unique():
            ld = loc_data[loc_data["frame"] == f]

            if not len(ld):
                continue
            im = sim.simulate_gauss(ims[0].shape[::-1], ld[["x", "y"]],
                                    ld["mass"], 1.0, mass=True,
                                    engine="python")
            ims[f] += im
        if ret_type == "list":
            yield ims, False
            return

        file = tmp_path / "test.tif"
        with tifffile.TiffWriter(file) as wrt:
            for i in ims:
                # If not setting contiguous=True, PIMS will fail
                wrt.write(i, contiguous=True)
        if ret_type == "pims":
            ret = pims.open(str(file))
            yield ret, True
            ret.close()
            return
        if ret_type == "ImageSequence":
            ret = io.ImageSequence(file).open()
            yield ret, True
            ret.close()
            return

    @pytest.fixture(params=["list", "pims", "ImageSequence"])
    def image_sequence(self, request, tmp_path, loc_data):
        with self._make_image_sequence(
                request.param, tmp_path, loc_data) as ret:
            yield ret

    @pytest.fixture
    def roi_corners(self):
        return np.array([(20.0, 25.0), (110.0, 75.0)])

    @pytest.fixture(params=["vertices", "mpl.path", "PathROI"])
    def path_roi(self, request, roi_corners):
        verts = [(roi_corners[0, 0], roi_corners[0, 1]),
                 (roi_corners[1, 0], roi_corners[0, 1]),
                 (roi_corners[1, 0], roi_corners[1, 1]),
                 (roi_corners[0, 0], roi_corners[1, 1])]
        if request.param == "vertices":
            return verts
        if request.param == "mpl.path":
            return mpl.path.Path(verts)
        if request.param == "PathROI":
            return roi.PathROI(verts)

    def test_locate(self, image_sequence, loc_data):
        img, frame_meta = image_sequence
        res = loc.cg.locate(img[1], 4, 50, 500, bandpass=False)
        assert "size" in res
        res.drop(columns="size", inplace=True)  # Size is too unpredictable

        exp = loc_data[loc_data["frame"] == 1]
        if not frame_meta:
            exp = exp.drop(columns="frame")
        pd.testing.assert_frame_equal(
            res.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
            exp.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
            rtol=1e-4, atol=0.06)

    def test_locate_roi(self, image_sequence, loc_data, path_roi, roi_corners):
        img, frame_meta = image_sequence
        exp = loc_data[(loc_data["x"] > roi_corners[0, 0]) &
                       (loc_data["x"] < roi_corners[1, 0]) &
                       (loc_data["y"] > roi_corners[0, 1]) &
                       (loc_data["y"] < roi_corners[1, 1]) &
                       (loc_data["frame"] == 1)]

        if not frame_meta:
            exp = exp.drop(columns="frame")

        res = loc.cg.locate_roi(img[1], path_roi, 4, 50, 500, bandpass=False,
                                rel_origin=False)
        assert "size" in res
        res.drop(columns="size", inplace=True)  # Size is too unpredictable

        pd.testing.assert_frame_equal(
            res.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
            exp.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
            rtol=1e-4, atol=0.06)

    def test_locate_regression(self, regression_data, regression_img):
        # Compare to output of original implementation (regression test)
        frame = regression_img[0]
        res = loc.cg.locate(frame, **regression_options)

        exp = regression_data[regression_data["frame"] == 0]
        check_regression_result(res, exp, frame.shape)

    def test_batch(self, image_sequence, loc_data):
        # Test the high level locate function only for one model (2d), since
        # the lower level functions are all tested separately for all models
        img, frame_meta = image_sequence

        res = loc.cg.batch(img[1:], 4, 50, 500, bandpass=False)
        assert "size" in res
        res.drop(columns="size", inplace=True)  # Size is too unpredictable

        exp = loc_data
        if not frame_meta:
            exp["frame"] -= 1

        pd.testing.assert_frame_equal(
            res.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
            exp.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
            rtol=1e-4, atol=0.065)

    def test_batch_roi(self, image_sequence, loc_data, path_roi, roi_corners):
        img, frame_meta = image_sequence
        exp = loc_data[(loc_data["x"] > roi_corners[0, 0]) &
                       (loc_data["x"] < roi_corners[1, 0]) &
                       (loc_data["y"] > roi_corners[0, 1]) &
                       (loc_data["y"] < roi_corners[1, 1])].copy()
        if not frame_meta:
            exp["frame"] -= 1

        res = loc.cg.batch_roi(img[1:], path_roi, 4, 50, 500, bandpass=False,
                               rel_origin=False)
        assert "size" in res
        res.drop(columns="size", inplace=True)  # Size is too unpredictable

        pd.testing.assert_frame_equal(
            res.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
            exp.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
            rtol=1e-4, atol=0.65)

    def test_batch_regression(self, regression_data, regression_img):
        # Compare to output of original implementation (regression test)
        res = loc.cg.batch(regression_img, **regression_options)
        assert "frame" in res
        check_regression_result(res, regression_data, regression_img[0].shape)


class TestFind:
    @pytest.fixture
    def regression_local_max(self):
        with np.load(data_path / "cg_local_max_orig.npz") as orig:
            return {int(k): v for k, v in orig.items()}

    def test_local_maxima(self):
        img = np.zeros((80, 100))
        img[30, 40] = 300
        img[10, 20] = 100  # too dim
        img[3, 40] = 300  # right at the border
        img[20, 3] = 300
        img[76, 41] = 300
        img[21, 96] = 300
        img[2, 60] = 300  # too close to border
        img[30, 2] = 300
        img[77, 50] = 300
        img[30, 97] = 300
        img[50, 40] = 400  # two peaks close together
        img[50, 42] = 300
        img[40:42, 70:72] = 300  # one flat peak
        img[43, 70] = 300

        res = find.local_maxima(img, 3, 150)
        print(res)
        exp = np.array([[30, 40],
                        [3, 40],
                        [20, 3],
                        [76, 41],
                        [21, 96],
                        [50, 40],
                        [43, 70]])
        np.testing.assert_array_equal(res[np.argsort(res[:, 0])],
                                      exp[np.argsort(exp[:, 0])])

    def test_local_maxima_regression(self, regression_img,
                                     regression_local_max):
        for i, img in enumerate(regression_img):
            radius = regression_options["radius"]
            lm = find.local_maxima(img, radius,
                                   regression_options["signal_thresh"])
            expected = regression_local_max[i].T
            # The original implementation's `local_maxima` removed maxima that
            # were ``radius + 1`` pixels away from right/bottom edge, which is
            # not necessary; removing maxima ``radius`` or less pixels away
            # from  edges is sufficient.
            wrongly_removed = np.any(
                lm == np.array(img.shape) - radius - 1,
                axis=1,
            )
            np.testing.assert_allclose(lm[~wrongly_removed], expected)

    def test_find_regression(self, regression_img, regression_local_max):
        for i, img in enumerate(regression_img):
            radius = regression_options["radius"]
            lm = find.find(img, radius, regression_options["signal_thresh"])
            expected = regression_local_max[i].T[:, ::-1]
            # The original implementation's `local_maxima` removed maxima that
            # were ``radius + 1`` pixels away from right/bottom edge, which is
            # not necessary; removing maxima ``radius`` or less pixels away
            # from  edges is sufficient.
            wrongly_removed = np.any(
                lm == np.array(img.shape[::-1]) - radius - 1,
                axis=1,
            )
            np.testing.assert_allclose(lm[~wrongly_removed], expected)
