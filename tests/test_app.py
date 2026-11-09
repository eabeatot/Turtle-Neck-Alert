import json
import shutil
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
    prediction = c.post('/api/predict',json={'points':pts()})
    assert prediction.status_code==200
    assert prediction.json()['app_version'] == 'dev'
    assert prediction.json()['model_version'] == 'v1'
    assert c.post('/api/rollback/v1').status_code==200
def test_empty_rejected(tmp_path,monkeypatch):
    monkeypatch.setenv('POSTURE_DATA_DIR',str(tmp_path)); import importlib, app.ml as ml; importlib.reload(ml); import app.main as main; importlib.reload(main)
    assert TestClient(main.app).post('/api/train').status_code==400

def test_image_default_updates_existing_volume_without_deleting_user_data(tmp_path, monkeypatch):
    image_models = tmp_path / 'image-models'
    image_models.mkdir()
    source = __import__('pathlib').Path(__file__).parents[1] / 'data' / 'models'
    shutil.copy2(source / 'v1.joblib', image_models / 'v1.joblib')
    (image_models / 'v1.metrics.json').write_text(json.dumps({'version':'v1','accuracy':1.0}), encoding='utf8')
    (image_models / 'active.json').write_text(json.dumps({'version':'v1'}), encoding='utf8')
    monkeypatch.setenv('POSTURE_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('POSTURE_IMAGE_MODELS_DIR', str(image_models))
    import importlib, app.ml as ml
    importlib.reload(ml)
    ml.ensure_default_model()
    (ml.DATA).write_text('{"label":"good"}\n', encoding='utf8')

    shutil.copy2(source / 'v1.joblib', image_models / 'v2.joblib')
    (image_models / 'v2.metrics.json').write_text(json.dumps({'version':'v2','accuracy':0.9}), encoding='utf8')
    (image_models / 'active.json').write_text(json.dumps({'version':'v2'}), encoding='utf8')
    ml.ensure_default_model()

    import app.main as main
    importlib.reload(main)
    response = TestClient(main.app).get('/health')
    assert response.json()['active_model']['version'] == 'v2'
    assert (ml.MODELS / 'v1.joblib').exists()
    assert (ml.MODELS / 'v2.joblib').exists()
    assert ml.DATA.read_text(encoding='utf8') == '{"label":"good"}\n'

def test_running_container_keeps_its_model_when_shared_volume_changes(tmp_path, monkeypatch):
    """blue-green 교체 중 새 컨테이너가 공유 볼륨의 active.json을 바꿔도 기존 컨테이너는 자기 모델로 계속 응답해야 한다."""
    data = tmp_path / 'data'; (data / 'models').mkdir(parents=True)
    source = __import__('pathlib').Path(__file__).parents[1] / 'models'
    shutil.copy2(source / 'v1.joblib', data / 'models' / 'v1.joblib')
    (data / 'active.json').write_text(json.dumps({'version':'v1','accuracy':1.0}), encoding='utf8')
    monkeypatch.setenv('POSTURE_DATA_DIR', str(data))
    import importlib, app.ml as ml; importlib.reload(ml); import app.main as main; importlib.reload(main)
    c = TestClient(main.app)
    assert c.post('/api/predict', json={'points':pts()}).json()['model_version'] == 'v1'
    # 다른 컨테이너가 공유 볼륨을 새 모델로 바꾼 상황 (기존 이미지가 못 읽는 모델 파일이라고 가정)
    (data / 'models' / 'v9.joblib').write_bytes(b'not a model for this image')
    (data / 'active.json').write_text(json.dumps({'version':'v9','accuracy':1.0}), encoding='utf8')
    r = c.post('/api/predict', json={'points':pts()})
    assert r.status_code == 200 and r.json()['model_version'] == 'v1'
