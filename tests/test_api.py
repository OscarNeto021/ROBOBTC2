from trading_api.src.main import app


def test_health_endpoint():
    client = app.test_client()
    resp = client.get('/api/trading/health')
    assert resp.status_code == 200
    assert 'status' in resp.get_json()


def test_signal_endpoint():
    client = app.test_client()
    resp = client.get('/api/trading/signal')
    assert resp.status_code == 200
    assert 'signal' in resp.get_json()
