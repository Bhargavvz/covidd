"""
Tests for data pipeline: preprocessing, augmentation, lung segmentation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


class TestPreprocessing:
    """Test CT preprocessing pipeline."""

    def test_normalize(self):
        from data.preprocessing import CTPreprocessor

        preprocessor = CTPreprocessor()
        arr = np.random.uniform(-1000, 1000, (64, 64, 64)).astype(np.float32)
        normalized = preprocessor.normalize(arr)

        assert normalized.min() >= -0.01
        assert normalized.max() <= 1.01
        assert normalized.dtype == np.float32

    def test_pad_or_crop_larger(self):
        from data.preprocessing import CTPreprocessor

        preprocessor = CTPreprocessor(target_size=(32, 32, 32))
        arr = np.random.randn(48, 48, 48).astype(np.float32)
        result = preprocessor.pad_or_crop(arr)

        assert result.shape == (32, 32, 32)

    def test_pad_or_crop_smaller(self):
        from data.preprocessing import CTPreprocessor

        preprocessor = CTPreprocessor(target_size=(64, 64, 64))
        arr = np.random.randn(32, 32, 32).astype(np.float32)
        result = preprocessor.pad_or_crop(arr)

        assert result.shape == (64, 64, 64)

    def test_pad_or_crop_exact(self):
        from data.preprocessing import CTPreprocessor

        preprocessor = CTPreprocessor(target_size=(32, 32, 32))
        arr = np.random.randn(32, 32, 32).astype(np.float32)
        result = preprocessor.pad_or_crop(arr)

        assert result.shape == (32, 32, 32)
        np.testing.assert_array_equal(result, arr)


class TestLungSegmentation:
    """Test lung segmentation module."""

    def test_segment_synthetic(self):
        from data.lung_segmentation import LungSegmenter

        segmenter = LungSegmenter(threshold=0.3, min_lung_volume=10)

        # Create a simple synthetic volume with "lung-like" region
        vol = np.ones((32, 32, 32), dtype=np.float32) * 0.8  # Body
        vol[8:24, 8:24, 8:24] = 0.1  # "Lung" region (low intensity)

        mask = segmenter.segment(vol, is_normalized=True)

        assert mask.shape == vol.shape
        assert mask.dtype == np.uint8
        assert mask.max() <= 1

    def test_compute_lung_statistics(self):
        from data.lung_segmentation import LungSegmenter

        segmenter = LungSegmenter()

        vol = np.random.randn(32, 32, 32).astype(np.float32)
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[8:24, 8:24, 8:24] = 1

        stats = segmenter.compute_lung_statistics(vol, mask)

        assert "mean_density" in stats
        assert "volume_ratio" in stats
        assert "upper_mean_density" in stats


class TestAugmentation:
    """Test registration augmentation."""

    def test_augmentor_shapes(self):
        from data.augmentation import RegistrationAugmentor

        augmentor = RegistrationAugmentor(
            affine_prob=1.0,
            elastic_prob=0.0,
            flip_prob=0.0,
            intensity_prob=0.0,
            seed=42,
        )

        moving = np.random.randn(32, 32, 32).astype(np.float32)
        fixed = np.random.randn(32, 32, 32).astype(np.float32)

        aug_moving, aug_fixed = augmentor(moving, fixed)

        assert aug_moving.shape == (32, 32, 32)
        assert aug_fixed.shape == (32, 32, 32)

    def test_no_augmentation(self):
        from data.augmentation import RegistrationAugmentor

        augmentor = RegistrationAugmentor(
            affine_prob=0.0,
            elastic_prob=0.0,
            flip_prob=0.0,
            intensity_prob=0.0,
        )

        moving = np.random.randn(32, 32, 32).astype(np.float32)
        fixed = np.random.randn(32, 32, 32).astype(np.float32)

        aug_moving, aug_fixed = augmentor(moving, fixed)

        np.testing.assert_array_equal(aug_moving, moving)
        np.testing.assert_array_equal(aug_fixed, fixed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
