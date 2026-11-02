import json
from fastapi.testclient import TestClient
def pts(lean=0):
    a=[{"index":i,"x":.5,"y":.3+i*.01+lean,"z":0} for i in range(33)]; a[11]={"index":11,"x":.4,"y":.5,"z":0}; a[12]={"index":12,"x":.6,"y":.5,"z":0}; return a
def test_cycle(tmp_path,monkeypatch):
    monkeypatch.setenv('POSTURE_DATA_DIR',str(tmp_path))
    import importlib, app.ml as ml; importlib.reload(ml); import app.main as main; importlib.reload(main)
    c=TestClient(main.app)
    assert c.get('/health').status_code==200
    for label,lean in [('good',0),('bad',.5)]: assert c.post('/api/data',json={'label':label,'samples':[pts(lean+i*.01) for i in range(12)]}).status_code==200
    x=c.post('/api/train'); assert x.status_code==200 and x.json()['version']=='v1'
    assert c.post('/api/predict',json={'points':pts()}).status_code==200
    assert c.post('/api/rollback/v1').status_code==200
def test_empty_rejected(tmp_path,monkeypatch):
    monkeypatch.setenv('POSTURE_DATA_DIR',str(tmp_path)); import importlib, app.ml as ml; importlib.reload(ml); import app.main as main; importlib.reload(main)
    assert TestClient(main.app).post('/api/train').status_code==400
