import pytest, io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from PIL import Image
 
 
def _jpeg_bytes() -> io.BytesIO:
    img = Image.new('RGB', (224, 224), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf
 
 
@pytest.fixture
def client():
    probs                        = MagicMock()
    probs.top1                   = 1          # Pneumonia
    probs.top1conf               = 0.88
    probs.data.tolist.return_value = [0.05, 0.88, 0.07]
 
    mock_result       = MagicMock()
    mock_result.probs = probs
 
    with patch('app.load_model') as mock_load:
        mock_model = MagicMock(return_value=[mock_result])
        mock_load.return_value = mock_model
        from app import app
        yield TestClient(app)
 
 
def test_health_ok(client):
    res = client.get('/health')
    assert res.status_code == 200
    assert res.json()['status'] == 'ok'
 
 
def test_predict_success(client):
    res = client.post('/predict', files={'file': ('xray.jpg', _jpeg_bytes(), 'image/jpeg')})
    assert res.status_code == 200
    data = res.json()
    assert 'predicted_class' in data
    assert 'confidence'      in data
    assert 'probabilities'   in data
 
 
def test_predict_rejects_bmp(client):
    res = client.post('/predict', files={'file': ('scan.bmp', b'fake', 'image/bmp')})
    assert res.status_code == 400
