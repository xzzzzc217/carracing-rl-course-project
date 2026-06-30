from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_frame(
    frame: np.ndarray,
    image_size: int = 64,
    crop_bottom: int = 12,
    grayscale: bool = True,
) -> np.ndarray:
    """Crop, grayscale, resize, and normalize one CarRacing RGB frame.

    Returns a channel-first float32 array in [0, 1].
    """
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected an RGB frame with shape (H, W, 3), got {frame.shape}.")

    image = np.asarray(frame)
    if np.issubdtype(image.dtype, np.floating) and image.max(initial=0.0) <= 1.0:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if crop_bottom > 0:
        if crop_bottom >= image.shape[0]:
            raise ValueError("crop_bottom must be smaller than the frame height.")
        image = image[: image.shape[0] - crop_bottom, :, :]

    if grayscale:
        gray = (
            0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]
        ).astype(np.uint8)
        pil_image = Image.fromarray(gray, mode="L")
        pil_image = pil_image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        processed = np.asarray(pil_image, dtype=np.float32) / 255.0
        return processed[None, :, :]

    pil_image = Image.fromarray(image, mode="RGB")
    pil_image = pil_image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    processed = np.asarray(pil_image, dtype=np.float32) / 255.0
    return np.transpose(processed, (2, 0, 1))


def encode_observation(observation: np.ndarray) -> np.ndarray:
    """Convert a normalized observation to uint8 storage format for replay."""
    if observation.dtype == np.uint8:
        return observation
    return np.clip(observation * 255.0, 0, 255).astype(np.uint8)


def decode_observation(observation: np.ndarray) -> np.ndarray:
    """Convert replay storage format back to normalized float32 tensors."""
    if observation.dtype == np.float32:
        return observation
    return observation.astype(np.float32) / 255.0
