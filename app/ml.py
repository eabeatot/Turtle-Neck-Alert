import json, math, os
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

ROOT = Path(os.getenv("POSTURE_DATA_DIR", "data")); ROOT.mkdir(exist_ok=True)
DATA = ROOT / "samples.jsonl"; MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
ACTIVE = ROOT / "active.json"
MIN_ACCURACY = 0.70

def now(): return datetime.now(timezone.utc).isoformat()
def feature_vector(points):
    """MediaPipe landmark list/dict -> shoulder-normalized x,y,z + head/neck features."""
    if isinstance(points, dict): points = points.get("landmarks", points.get("points", []))
    p = {int(x["index"] if isinstance(x, dict) else i): (float(x["x"]), float(x["y"]), float(x.get("z", 0) if isinstance(x, dict) else (x[2] if len(x)>2 else 0))) for i,x in enumerate(points)}
    if 11 not in p or 12 not in p: raise ValueError("어깨 좌표(11, 12)가 필요합니다")
    sx,sy,sz=p[11]; tx,ty,tz=p[12]; scale=max(math.hypot(sx-tx, sy-ty), 1e-6); cx=(sx+tx)/2; cy=(sy+ty)/2; cz=(sz+tz)/2
    ids=sorted(p); out=[]
    for i in ids: out += [(p[i][0]-cx)/scale,(p[i][1]-cy)/scale,(p[i][2]-cz)/scale]
    # stable fixed-length posture cues; missing points become zero
    for i in (0,7,8): out += [((p[i][0]-cx)/scale if i in p else 0),((p[i][1]-cy)/scale if i in p else 0)]
    return np.asarray(out, dtype=float)

def save_samples(label, samples, session_id=None):
    if not samples: raise ValueError("저장할 좌표가 없습니다")
    sid=session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    with DATA.open("a", encoding="utf8") as f:
        for pts in samples: f.write(json.dumps({"session_id":sid,"timestamp":now(),"label":label,"points":pts}, ensure_ascii=False)+"\n")
    return {"session_id":sid,"count":len(samples)}

def read_rows():
    if not DATA.exists(): return []
    return [json.loads(x) for x in DATA.read_text(encoding="utf8").splitlines() if x.strip()]

def train_model():
    rows=read_rows();
    if len(rows)<4 or len({r["label"] for r in rows})<2: raise ValueError("두 라벨의 데이터가 각각 필요합니다")
    X=np.array([feature_vector(r["points"]) for r in rows]); y=np.array([r["label"] for r in rows])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)); model.fit(Xtr,ytr); acc=float(accuracy_score(yte,model.predict(Xte)))
    versions=[int(p.stem[1:]) for p in MODELS.glob("v*.joblib") if p.stem[1:].isdigit()]; version=max(versions or [0])+1
    metrics={"version":f"v{version}","accuracy":acc,"trained_at":now(),"sample_count":len(rows),"activated":False,"reason":""}
    joblib.dump(model, MODELS/f"v{version}.joblib"); (MODELS/f"v{version}.metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf8")
    current=json.loads(ACTIVE.read_text()) if ACTIVE.exists() else None; old=current.get("accuracy",0) if current else 0
    if acc>=MIN_ACCURACY and acc>=old:
        metrics["activated"]=True; ACTIVE.write_text(json.dumps({"version":metrics["version"],"accuracy":acc},ensure_ascii=False),encoding="utf8")
    else: metrics["reason"]=f"검증 게이트 미통과: 정확도 {acc:.3f}, 기준/현재 모델 {max(MIN_ACCURACY,old):.3f}"
    (MODELS/f"v{version}.metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf8")
    return metrics

def active_model():
    if not ACTIVE.exists(): return None
    a=json.loads(ACTIVE.read_text()); path=MODELS/f"{a['version']}.joblib"
    return (a,joblib.load(path)) if path.exists() else None
def activate(version):
    m=MODELS/f"{version}.metrics.json"
    if not m.exists(): raise ValueError("존재하지 않는 모델 버전입니다")
    x=json.loads(m.read_text()); ACTIVE.write_text(json.dumps({"version":version,"accuracy":x["accuracy"]}),encoding="utf8"); return x

def delete_data():
    if DATA.exists(): DATA.unlink()
    return {"deleted":"data"}

def delete_models():
    for p in MODELS.glob("v*"):
        if p.is_file(): p.unlink()
    if ACTIVE.exists(): ACTIVE.unlink()
    return {"deleted":"models"}
