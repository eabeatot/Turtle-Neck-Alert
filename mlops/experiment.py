"""MLOps 개선 실험: v1(원본 좌표) vs v2(자세 특징 전처리)

규칙
- 세션 단위 분할: 같은 세션의 프레임은 거의 같아서, 프레임 단위로 나누면 정확도가 부풀려진다.
- 튜닝용 = 기존 v1 학습 세션(11/02, 2개) + 11/09 첫 바른자세·첫 나쁜자세 세션(2개)
- 평가용 = 11/09 나머지 6개 세션(바른 3, 나쁜 3) → 튜닝·모델 선택에 사용하지 않음
- 개선 시도 최대 2회, 시도 간 선택은 튜닝 세션 Leave-One-Session-Out 정확도로만 결정
실행: 저장소 루트에서  python mlops/experiment.py   (models/v2.* 생성)
"""
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["POSTURE_DATA_DIR"] = tempfile.mkdtemp()  # 서비스 data/를 건드리지 않음

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app import ml
from app.features import posture_features, posture_features_2d

DATA = ROOT / "mlops" / "data"
TUNE_NEW = {"20261109-120353", "20261109-120444"}  # 11/09 첫 바른자세, 첫 나쁜자세 세션


def load(name):
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf8").splitlines() if l.strip()]


def xy(rows):
    return (np.array([ml.feature_vector(r["points"]) for r in rows]),
            np.array([r["label"] for r in rows]),
            np.array([r["session_id"] for r in rows]))


old, new = load("sessions_20261102.jsonl"), load("sessions_20261109.jsonl")
tune_rows = old + [r for r in new if r["session_id"] in TUNE_NEW]
eval_rows = [r for r in new if r["session_id"] not in TUNE_NEW]
Xt, yt, st = xy(tune_rows)
Xe, ye, se = xy(eval_rows)

CANDIDATES = {
    "baseline (원본 좌표, v1 방식)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "시도1: 자세 특징 8개": lambda: make_pipeline(FunctionTransformer(posture_features), StandardScaler(), LogisticRegression(max_iter=1000)),
    "시도2: 2D 특징 6개 + 강한 규제(C=0.1)": lambda: make_pipeline(FunctionTransformer(posture_features_2d), StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
}


def loso(make):
    accs = []
    for sid in dict.fromkeys(st):
        tr = st != sid
        m = make().fit(Xt[tr], yt[tr])
        accs.append(float((m.predict(Xt[~tr]) == yt[~tr]).mean()))
    return float(np.mean(accs)), accs


def report(model, X, y, s):
    p = model.predict(X)
    return {
        "accuracy": round(float(accuracy_score(y, p)), 3),
        "f1_bad": round(float(f1_score(y, p, pos_label="bad")), 3),
        "mean_confidence": round(float(model.predict_proba(X).max(1).mean()), 3),
        "confusion(행=정답 good,bad / 열=예측 good,bad)": confusion_matrix(y, p, labels=["good", "bad"]).tolist(),
        "per_session": {sid: round(float((p[s == sid] == y[s == sid]).mean()), 3) for sid in dict.fromkeys(s)},
    }


print(f"튜닝 세션 {len(set(st))}개 / {len(yt)}프레임,  평가 세션 {len(set(se))}개 / {len(ye)}프레임\n")
print("[1] 튜닝 세션 Leave-One-Session-Out (모델 선택 기준)")
scores = {}
for name, make in CANDIDATES.items():
    mean, accs = loso(make)
    scores[name] = mean
    print(f"  {name:<34} 평균 {mean:.3f}   세션별 {[round(a, 3) for a in accs]}")
best = max([k for k in CANDIDATES if k.startswith("시도")], key=scores.get)
print(f"  → 선택: {best}\n")

v1 = joblib.load(ROOT / "models" / "v1.joblib")
v2 = CANDIDATES[best]().fit(Xt, yt)
r1, r2 = report(v1, Xe, ye, se), report(v2, Xe, ye, se)
print("[2] 최종 평가 세트 (튜닝에 사용하지 않음)")
print(f"  {'지표':<16}{'v1 (변경 전)':>14}{'v2 (변경 후)':>14}")
for k in ("accuracy", "f1_bad", "mean_confidence"):
    print(f"  {k:<16}{r1[k]:>14}{r2[k]:>14}")
ck = "confusion(행=정답 good,bad / 열=예측 good,bad)"
print(f"  confusion v1 {r1[ck]}   v2 {r2[ck]}   (행=정답 good,bad / 열=예측 good,bad)")
print("  세션별 정확도")
labels = {r["session_id"]: r["label"] for r in eval_rows}
for sid in r1["per_session"]:
    print(f"    {sid} ({labels[sid]:<4}) v1 {r1['per_session'][sid]:.3f} → v2 {r2['per_session'][sid]:.3f}")

joblib.dump(v2, ROOT / "models" / "v2.joblib")
metrics = {
    "version": "v2", "accuracy": r2["accuracy"], "trained_at": datetime.now(timezone.utc).isoformat(),
    "sample_count": int(len(yt)), "activated": True, "reason": "",
    "change": best, "evaluation": {"eval_sessions": sorted(set(se.tolist())), "v1": r1, "v2": r2},
}
(ROOT / "models" / "v2.metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
print("\nmodels/v2.joblib, models/v2.metrics.json 저장 완료")
