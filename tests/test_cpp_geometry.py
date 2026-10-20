import os
import subprocess
from pathlib import Path
import numpy as np
import pytest
from poseguard.geometry import decode, preprocess
from poseguard.types import Letterbox
from poseguard.image_io import write_image

CHECK = Path(os.environ.get("POSEGUARD_GEOMETRY_CHECK", "build-gpu/cpp/Release/pose_geometry_check.exe" if os.name == "nt" else "build/cpp/pose_geometry_check"))
pytestmark = pytest.mark.skipif(os.environ.get("POSEGUARD_TEST_GPU") != "1", reason="Requires compiled C++ geometry check")


@pytest.mark.parametrize("shape", [(100, 100), (321, 777), (2160, 3840), (640, 480)])
def test_native_preprocessing_matches_python(tmp_path, shape):
    image = np.random.default_rng(1).integers(0, 256, (*shape, 3), dtype=np.uint8)
    source, result = tmp_path / "image.png", tmp_path / "tensor.bin"
    write_image(source, image)
    subprocess.run([str(CHECK), "preprocess", str(source), str(result), "640"], check=True, timeout=15)
    tensor, _ = preprocess(image)
    np.testing.assert_array_equal(np.fromfile(result, np.float32).reshape(tensor.shape), tensor)


@pytest.mark.parametrize("meta", [Letterbox(1,0,0,100,100,100), Letterbox(.5,0,20,200,120,100)])
def test_native_nms_border_nan_ties_and_confidence_clipping(tmp_path, meta):
    rows = np.zeros((8, 56), np.float32)
    rows[:, :5] = [[-100,50,400,80,.9], [25,50,150,80,.8], [50,50,40,40,.7],
                  [50,50,40,40,.7], [-300,50,10,10,.99], [90,90,20,20,.6],
                  [50,50,40,40,.95], [50,50,0,10,.99]]
    rows[:, 5:] = np.tile([50,50,1.2],17)
    rows[6,5] = np.nan
    raw = rows.T[None].copy()
    source, result = tmp_path / "raw.bin", tmp_path / "decoded.bin"
    raw.tofile(source)
    subprocess.run([str(CHECK), "decode", str(source), str(result), str(meta.scale), str(meta.left),
                    str(meta.top), str(meta.width), str(meta.height)], check=True, timeout=15)
    expected = np.asarray([np.r_[d.bbox,d.confidence,d.keypoints.ravel()] for d in decode(raw,meta)], np.float32)
    actual = np.fromfile(result,np.float32).reshape(-1,56)
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-6)
