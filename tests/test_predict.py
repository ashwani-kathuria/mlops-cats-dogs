import torch
from PIL import Image
from src.inference.predict import preprocess_image, predict
import pytest


# ------------------------------------------------
# Test 1: Preprocess output shape
# ------------------------------------------------
def test_preprocess_image_shape(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (300, 300))
    img.save(img_path)

    tensor = preprocess_image(str(img_path))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)


# ------------------------------------------------
# Test 2: Preprocess handles RGB conversion
# ------------------------------------------------
def test_preprocess_converts_to_rgb(tmp_path):
    img_path = tmp_path / "gray.jpg"

    # create grayscale image
    img = Image.new("L", (300, 300))
    img.save(img_path)

    tensor = preprocess_image(str(img_path))

    # Should still return 3 channels after conversion
    assert tensor.shape[1] == 3


# ------------------------------------------------
# Dummy model for inference test
# ------------------------------------------------
class DummyModel(torch.nn.Module):
    def forward(self, x):
        # return fake logits for 2 classes
        return torch.tensor([[0.1, 0.9]])


# ------------------------------------------------
# Test 3: predict() returns class label
# ------------------------------------------------
def test_predict_returns_valid_label(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (300, 300))
    img.save(img_path)

    dummy_model = DummyModel()

    result = predict(dummy_model, str(img_path))

    assert result in ["Cat", "Dog"]


# ------------------------------------------------
# Test 4: predict raises error for missing file
# ------------------------------------------------
def test_predict_invalid_path():
    dummy_model = DummyModel()

    with pytest.raises(Exception):
        predict(dummy_model, "non_existent.jpg")
