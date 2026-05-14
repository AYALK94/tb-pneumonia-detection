import pytest
from src.utils import validate_image, bytes_to_image, get_confidence_label
from PIL import Image
import io
 
 
def _make_jpeg() -> bytes:
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()
 
 
def test_validate_image_valid():
    assert validate_image('xray.jpg', _make_jpeg()) is True
 
 
def test_validate_image_invalid_extension():
    with pytest.raises(ValueError, match='Unsupported file type'):
        validate_image('xray.bmp', _make_jpeg())
 
 
def test_validate_image_too_large():
    big = b'x' * (11 * 1024 * 1024)   # 11 MB
    with pytest.raises(ValueError, match='File too large'):
        validate_image('xray.jpg', big)
 
 
def test_bytes_to_image_returns_rgb():
    img = bytes_to_image(_make_jpeg())
    assert isinstance(img, Image.Image)
    assert img.mode == 'RGB'
 
 
def test_get_confidence_label_very_high():
    assert get_confidence_label(95) == 'Very High'
 
def test_get_confidence_label_high():
    assert get_confidence_label(80) == 'High'
 
def test_get_confidence_label_moderate():
    assert get_confidence_label(60) == 'Moderate'
 
def test_get_confidence_label_low():
    assert get_confidence_label(40) == 'Low'
