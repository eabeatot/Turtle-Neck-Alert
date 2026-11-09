FROM python:3.14-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 POSTURE_DATA_DIR=/app/data
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY models ./models
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh && python -c "from app import ml; ml.ensure_default_model(); assert ml.active_model() is not None; p=[{'index':i,'x':.5,'y':.3+i*.01,'z':0} for i in range(33)]; print(ml.active_model()[1].predict([ml.feature_vector(p)]))"
ARG COMMIT_SHA=dev
ENV COMMIT_SHA=$COMMIT_SHA
VOLUME ["/app/data"]
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=12 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
