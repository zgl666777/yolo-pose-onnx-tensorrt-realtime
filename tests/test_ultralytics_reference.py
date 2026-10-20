import numpy as np
import pytest
from poseguard.geometry import preprocess


@pytest.mark.parametrize("shape", [(360, 640), (721, 1281), (600, 337), (1080, 810)])
def test_preprocessing_matches_ultralytics_letterbox(shape):
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox
    rng = np.random.default_rng(13)
    image = rng.integers(0, 256, (*shape, 3), np.uint8)
    reference = LetterBox(new_shape=(640, 640), auto=False, scaleup=True)(image=image)
    expected = np.ascontiguousarray(reference[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    actual, _ = preprocess(image, 640)
    np.testing.assert_array_equal(actual, expected)
