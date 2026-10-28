FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/harness-cache
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[local]" && useradd --create-home --uid 1000 hc \
    && mkdir -p /data /runs /archive /scenarios && chown hc:hc /data /runs /archive

ARG HC_CODE_SHA=unknown
ENV HC_CODE_SHA=${HC_CODE_SHA}

USER hc
WORKDIR /home/hc
# The container stays up; experiments are driven with `docker exec`.
CMD ["sleep", "infinity"]
