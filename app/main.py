from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import ml
app=FastAPI(title="Posture Helper"); app.mount("/static",StaticFiles(directory="app/static"),name="static")
class Samples(BaseModel): label:str; samples:list
class Predict(BaseModel): points:list
@app.get("/")
def index(): return FileResponse("app/static/index.html")
@app.get("/favicon.ico")
def favicon(): return FileResponse("app/static/favicon.svg", media_type="image/svg+xml")
@app.get("/health")
def health():
    a=ml.active_model(); rows=ml.read_rows(); return {"status":"ok","active_model":a[0] if a else None,"samples":len(rows),"label_counts":{"good":sum(r.get("label")=="good" for r in rows),"bad":sum(r.get("label")=="bad" for r in rows)}}
@app.post("/api/data")
def data(x:Samples):
    try:return ml.save_samples(x.label,x.samples)
    except ValueError as e: raise HTTPException(400,str(e))
@app.delete("/api/data")
def delete_data(): return ml.delete_data()
@app.delete("/api/models")
def delete_models(): return ml.delete_models()
@app.post("/api/train")
def train():
    try:return ml.train_model()
    except ValueError as e: raise HTTPException(400,str(e))
@app.post("/api/predict")
def predict(x:Predict):
    a=ml.active_model()
    if not a: raise HTTPException(503,"활성 모델이 없습니다. 먼저 데이터를 수집하고 학습하세요")
    try:
        f=ml.feature_vector(x.points).reshape(1,-1); m=a[1]; probs=m.predict_proba(f)[0]; i=int(m.classes_.tolist().index(m.predict(f)[0])); return {"label":str(m.predict(f)[0]),"confidence":float(probs[i])}
    except Exception as e: raise HTTPException(400,str(e))
@app.post("/api/rollback/{version}")
def rollback(version:str):
    try:return ml.activate(version)
    except ValueError as e: raise HTTPException(404,str(e))
