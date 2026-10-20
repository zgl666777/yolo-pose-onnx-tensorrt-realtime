import numpy as np
import pytest
from poseguard.geometry import preprocess, decode, nms


def test_letterbox_rgb_normalization_layout_and_round_trip():
    image = np.zeros((360, 640, 3), np.uint8)
    image[:, :] = [10, 20, 30]
    tensor, meta = preprocess(image, 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    assert (meta.left, meta.top, meta.scale) == (0, 140, 1)
    np.testing.assert_allclose(tensor[0, :, 200, 20], [30 / 255, 20 / 255, 10 / 255])
    np.testing.assert_allclose(tensor[0, :, 0, 0], 114 / 255)
    row = np.zeros(56, np.float32)
    row[:5] = [320, 320, 100, 200, 0.9]
    row[5:] = np.tile([320, 320, 0.8], 17)
    det = decode(row.reshape(1, 56, 1), meta)[0]
    np.testing.assert_allclose(det.bbox, [270, 80, 370, 280])
    np.testing.assert_allclose(det.keypoints[:, :2], np.tile([320, 180], (17, 1)))


@pytest.mark.parametrize("shape", [(721, 1280), (600, 337), (13, 4097)])
def test_letterbox_asymmetric_padding(shape):
    tensor, meta = preprocess(np.zeros((*shape, 3), np.uint8), 640)
    assert tensor.shape == (1, 3, 640, 640)
    row = np.zeros(56, np.float32)
    row[:5] = [shape[1] * meta.scale / 2 + meta.left,
               shape[0] * meta.scale / 2 + meta.top,
               shape[1] * meta.scale, shape[0] * meta.scale, 0.9]
    row[5:] = np.tile([row[0], row[1], 0.9], 17)
    det = decode(row.reshape(1, 56, 1), meta)[0]
    np.testing.assert_allclose(det.bbox, [0, 0, shape[1], shape[0]], atol=0.001)


def test_nms_suppresses_duplicates_but_preserves_distinct_people():
    boxes = np.array([[0, 0, 100, 100], [1, 1, 101, 101], [200, 200, 300, 300]])
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45) == [0, 2]


def test_decode_rejects_incompatible_model():
    _, meta = preprocess(np.zeros((200, 300, 3), np.uint8))
    with pytest.raises(ValueError, match="56 channels"):
        decode(np.zeros((1, 6, 100)), meta)


def test_decode_ignores_nan_and_negative_dimensions():
    _, meta = preprocess(np.zeros((640, 640, 3), np.uint8))
    rows = np.zeros((3, 56), np.float32)
    rows[:, :5] = [100, 100, 50, 50, 0.9]
    rows[0, 8] = np.nan
    rows[1, 2] = -20
    assert len(decode(rows.T[None], meta)) == 1

