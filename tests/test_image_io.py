import numpy as np
from poseguard.image_io import read_image, write_image


def test_unicode_image_path_roundtrip(tmp_path):
    path = tmp_path / "\u59ff\u6001.png"
    original = np.full((13, 17, 3), 127, np.uint8)
    write_image(path, original)
    np.testing.assert_array_equal(read_image(path), original)
    assert read_image(tmp_path / "missing.png") is None
