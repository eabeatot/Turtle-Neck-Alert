#!/bin/sh
set -eu
python -c 'from app import ml; ml.ensure_default_model()'
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
