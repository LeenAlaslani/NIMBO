FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY app/web/package*.json ./
RUN npm ci --no-audit --no-fund
COPY app/web/index.html app/web/vite.config.js ./
COPY app/web/src ./src
COPY app/web/public/nimbo-brand-mark.png ./public/
RUN VITE_NIMBO_REVIEWER=1 npm run build

FROM python:3.11-slim-bookworm AS inputs
WORKDIR /source
RUN pip install --no-cache-dir huggingface-hub==0.36.2 pyarrow==23.0.1
COPY app/nimbo_reviewer_catalog.py app/nimbo_reviewer_catalog.py
COPY tools/prepare_nimbo_reviewer.py tools/prepare_nimbo_reviewer.py
RUN HF_HOME=/tmp/dataset-download HF_HUB_DISABLE_XET=1 python -m tools.prepare_nimbo_reviewer /opt/nimbo-inputs

FROM python:3.11-slim-bookworm AS runtime
WORKDIR /opt/nimbo
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=2 update \
    && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home nimbo \
    && install -d -o nimbo -g nimbo -m 0700 /var/tmp/nimbo
ARG TARGETARCH
ARG TORCH_VERSION=2.9.1
RUN if [ "$TARGETARCH" = amd64 ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}+cpu"; \
    else pip install --no-cache-dir "torch==${TORCH_VERSION}"; fi
COPY deploy/reviewer-requirements.txt .
RUN pip install --no-cache-dir -r reviewer-requirements.txt
COPY --from=inputs /opt/nimbo-inputs /opt/nimbo-inputs
COPY --from=frontend /build/dist /opt/nimbo/web
COPY app/nimbo_model_intake_v1.py app/nimbo_onnx_candidate_v1.py app/nimbo_heldout_validation_v1.py app/nimbo_package_v1.py app/nimbo_auto_pipeline_jobs_v1.py app/nimbo_reviewer*.py /opt/nimbo/app/
COPY tools/run_nimbo_reviewer_job.py tools/verify_nimbo_package_v1.py /opt/nimbo/tools/
COPY deploy/replay.json /opt/nimbo/replay.json
COPY deploy/LICENSE-Apache-2.0.txt /opt/nimbo/deploy/
ENV PYTHONDONTWRITEBYTECODE=1 NIMBO_INPUTS=/opt/nimbo-inputs NIMBO_STATE=/var/tmp/nimbo \
    NIMBO_WEB=/opt/nimbo/web NIMBO_REPLAY=/opt/nimbo/replay.json
USER 10001
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=4s CMD python -c 'import os,urllib.request; urllib.request.urlopen("http://127.0.0.1:"+os.getenv("PORT","10000")+"/healthz",timeout=3)'
CMD ["python", "-m", "app.nimbo_reviewer"]
