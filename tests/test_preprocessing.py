from __future__ import annotations

import numpy as np

from carracing_rl.preprocessing import decode_observation, encode_observation, preprocess_frame


def test_preprocess_frame_shape_range_and_dtype() -> None:
    frame = np.random.default_rng(0).integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    processed = preprocess_frame(frame, image_size=64, crop_bottom=12, grayscale=True)
    assert processed.shape == (1, 64, 64)
    assert processed.dtype == np.float32
    assert 0.0 <= float(processed.min()) <= float(processed.max()) <= 1.0


def test_replay_encoding_roundtrip_is_normalized() -> None:
    observation = np.random.default_rng(1).random((4, 64, 64), dtype=np.float32)
    encoded = encode_observation(observation)
    decoded = decode_observation(encoded)
    assert encoded.dtype == np.uint8
    assert decoded.dtype == np.float32
    assert decoded.shape == observation.shape
    assert np.max(np.abs(decoded - observation)) <= 1.0 / 255.0 + 1e-6
