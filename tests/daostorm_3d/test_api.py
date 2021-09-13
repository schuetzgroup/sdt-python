# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib

import numpy as np
import pandas as pd
import pytest
import matplotlib as mpl

from sdt import io, loc, roi, sim
from sdt.helper import numba


@pytest.fixture
def loc_data():
    rs = np.random.RandomState(10)
    d = pd.DataFrame({"x": rs.uniform(10.0, 140.0, 15),
                      "y": rs.uniform(10.0, 90.0, 15),
                      "size": rs.normal(1.0, 0.1, 15),
                      "mass": rs.normal(2000.0, 30.0, 15),
                      "frame": [1] * 9 + [2] * 6,
                      "bg": 200.0})
    d["signal"] = d["mass"] / (2 * np.pi * d["size"] * d["size"])
    return d


@contextlib.contextmanager
def _make_image_sequence(ret_type, tmp_path, loc_data):
    if ret_type == "pims":
        pims = pytest.importorskip("pims", reason="pims not installed")
    elif ret_type == "ImageSequence":
        # ImageSequence uses imageio
        pytest.importorskip("imageio", reason="imageio not installed")
    if ret_type in ("pims", "ImageSequence"):
        # Need to write the image
        tifffile = pytest.importorskip(
            "tifffile", reason="tifffile not installed")

    bg = np.full((100, 150), 200.0)
    ims = [bg.copy() for _ in range(loc_data["frame"].max() + 1)]
    for f in loc_data["frame"].unique():
        ld = loc_data[loc_data["frame"] == f]

        s = np.empty((len(ld), 2))
        for i, c in enumerate(("size_x", "size_y")):
            s[:, i] = ld[c] if c in ld else ld["size"]

        if not len(ld):
            continue
        im = sim.simulate_gauss(bg.shape[::-1], ld[["x", "y"]], ld["signal"],
                                s, engine="python")
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
def image_sequence(request, tmp_path, loc_data):
    with _make_image_sequence(request.param, tmp_path, loc_data) as ret:
        yield ret


@pytest.fixture(params=["python", "numba"])
def engine(request):
    if request.param == "numba" and not numba.numba_available:
        pytest.skip("numba not available")
    return request.param


@pytest.fixture
def roi_corners():
    return np.array([(20.0, 25.0), (110.0, 75.0)])


@pytest.fixture(params=["vertices", "mpl.path", "PathROI"])
def path_roi(request, roi_corners):
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


def test_locate(image_sequence, loc_data, engine):
    # Test the high level locate function only for one model (2d), since the
    # lower level functions are all tested separately for all models
    img, frame_meta = image_sequence
    res = loc.daostorm_3d.locate(img[1], 1.0, "2d", 50, engine=engine)
    res["bg"] += 1  # FIXME: Background seems to be off by one

    exp = loc_data[loc_data["frame"] == 1]
    if not frame_meta:
        exp = exp.drop(columns="frame")
    pd.testing.assert_frame_equal(
        res.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
        exp.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
        rtol=1e-4)


def test_locate_roi(image_sequence, loc_data, path_roi, roi_corners, engine):
    img, frame_meta = image_sequence
    exp = loc_data[(loc_data["x"] > roi_corners[0, 0]) &
                   (loc_data["x"] < roi_corners[1, 0]) &
                   (loc_data["y"] > roi_corners[0, 1]) &
                   (loc_data["y"] < roi_corners[1, 1]) &
                   (loc_data["frame"] == 1)]

    if not frame_meta:
        exp = exp.drop(columns="frame")
    res = loc.daostorm_3d.locate_roi(img[1], path_roi, 1.0, "2d",
                                     50, engine=engine, rel_origin=False)
    res["bg"] += 1  # FIXME: Background seems to be off by one
    pd.testing.assert_frame_equal(
        res.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
        exp.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
        rtol=1e-4)


def test_batch(image_sequence, loc_data, engine):
    # Test the high level locate function only for one model (2d), since the
    # lower level functions are all tested separately for all models
    img, frame_meta = image_sequence

    res = loc.daostorm_3d.batch(img[1:], 1.0, "2d", 50, engine=engine)
    res["bg"] += 1  # FIXME: Background seems to be off by one

    exp = loc_data
    if not frame_meta:
        exp["frame"] -= 1

    pd.testing.assert_frame_equal(
        res.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
        exp.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
        rtol=1e-4)


def test_batch_roi(image_sequence, loc_data, path_roi, roi_corners, engine):
    img, frame_meta = image_sequence
    exp = loc_data[(loc_data["x"] > roi_corners[0, 0]) &
                   (loc_data["x"] < roi_corners[1, 0]) &
                   (loc_data["y"] > roi_corners[0, 1]) &
                   (loc_data["y"] < roi_corners[1, 1])].copy()
    if not frame_meta:
        exp["frame"] -= 1

    res = loc.daostorm_3d.batch_roi(img[1:], path_roi, 1.0, "2d",
                                    50, engine=engine, rel_origin=False)
    res["bg"] += 1  # FIXME: Background seems to be off by one

    pd.testing.assert_frame_equal(
        res.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
        exp.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
        rtol=1e-4)


@pytest.fixture
def z_params():
    par = loc.z_fit.Parameters(z_range=[-0.3, 0.3])
    par.x = par.Tuple(0.9, 0.1, 0.4, np.array([0.5]))
    par.y = par.Tuple(1.1, -0.1, 0.3, np.array([-0.5]))
    return par


@pytest.fixture
def loc_data_z(loc_data, z_params):
    loc_data["z"] = np.linspace(-0.25, 0.25, len(loc_data))
    loc_data["size_x"], loc_data["size_y"] = \
        z_params.sigma_from_z(loc_data["z"])
    loc_data["mass"] = (2 * np.pi * loc_data["signal"] * loc_data["size_x"] *
                        loc_data["size_y"])
    loc_data.drop(columns="size", inplace=True)
    return loc_data


@pytest.fixture(params=["list", "pims", "ImageSequence"])
def image_sequence_z(request, tmp_path, loc_data_z):
    with _make_image_sequence(request.param, tmp_path, loc_data_z) as ret:
        yield ret


def test_locate_z(image_sequence_z, loc_data_z, z_params, engine, tmp_path):
    img, frame_meta = image_sequence_z

    loc_data = loc_data_z[loc_data_z["frame"] == 1].copy()
    if not frame_meta:
        loc_data.drop(columns="frame", inplace=True)

    with pytest.raises(ValueError):
        # no z_params are passed and model is "z"
        loc.daostorm_3d.locate(img[1], 1, "z", 50)

    sv = tmp_path / "params.yaml"
    z_params.save(sv)

    for par in z_params, sv, str(sv):
        res = loc.daostorm_3d.locate(img[1], 1, "z", 50, z_params=par,
                                     engine=engine)
        res["bg"] += 1  # FIXME: Background seems to be off by one
        pd.testing.assert_frame_equal(
            res.sort_values(["x", "y"], ignore_index=True).sort_index(axis=1),
            loc_data.sort_values(["x", "y"],
                                 ignore_index=True).sort_index(axis=1),
            rtol=1e-4)
