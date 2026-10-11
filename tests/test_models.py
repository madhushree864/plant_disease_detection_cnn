import unittest

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - allows the suite without optional TF
    tf = None

from src.models import build_custom_cnn, build_efficientnetb0, set_backbone_trainability


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class ModelFactoryTests(unittest.TestCase):
    def test_models_have_expected_output_shape(self):
        self.assertEqual(build_custom_cnn(5).output_shape, (None, 5))
        self.assertEqual(
            build_efficientnetb0(5, weights=None).output_shape, (None, 5)
        )

    def test_custom_cnn_has_no_efficientnet_layers(self):
        self.assertFalse(any("efficientnet" in layer.name.lower() for layer in build_custom_cnn(5).layers))

    def test_freeze_helper_only_unfreezes_upper_backbone_layers(self):
        model = build_efficientnetb0(5, weights=None)
        backbone = model.get_layer("efficientnetb0_backbone")
        set_backbone_trainability(model, trainable_layers=3)
        self.assertEqual(sum(layer.trainable for layer in backbone.layers), 3)
        self.assertTrue(all(not layer.trainable for layer in backbone.layers[:-3]))
        self.assertTrue(all(layer.trainable for layer in backbone.layers[-3:]))


if __name__ == "__main__":
    unittest.main()
