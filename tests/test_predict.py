import pytest
from unittest.mock import MagicMock
from PIL import Image
from src.predict import predict_image
 
 
def _mock_model(top1: int = 0, top1conf: float = 0.95) -> MagicMock:
    probs            = MagicMock()
    probs.top1       = top1
    probs.top1conf   = top1conf
    probs.data.tolist.return_value = [0.95, 0.03, 0.02]
 
    result       = MagicMock()
    result.probs = probs
 
    model        = MagicMock(return_value=[result])
    return model
 
 
def test_predict_normal_class():
    result = predict_image(_mock_model(top1=0), Image.new('RGB', (224, 224)))
    assert result['predicted_class'] == 'Normal'
 
 
def test_predict_pneumonia_class():
    result = predict_image(_mock_model(top1=1), Image.new('RGB', (224, 224)))
    assert result['predicted_class'] == 'Pneumonia'
 
 
def test_confidence_is_percentage():
    result = predict_image(_mock_model(top1conf=0.87), Image.new('RGB', (224, 224)))
    assert 0 <= result['confidence'] <= 100
 
 
def test_all_three_classes_in_probabilities():
    result = predict_image(_mock_model(), Image.new('RGB', (224, 224)))
    for cls in ['Normal', 'Pneumonia', 'TB']:
        assert cls in result['probabilities']
 
 
def test_raises_when_probs_none():
    result       = MagicMock()
    result.probs = None
    model        = MagicMock(return_value=[result])
    with pytest.raises(ValueError, match='probabilities'):
        predict_image(model, Image.new('RGB', (224, 224)))
