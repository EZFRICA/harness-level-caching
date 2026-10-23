FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/cama
COPY pyproject.toml ./
COPY src ./src
RUN pip install . && useradd --create-home --uid 1000 cama \
    && mkdir -p /data /runs /archive /scenarios && chown cama:cama /data /runs /archive

ARG CAMA_CODE_SHA=unknown
ENV CAMA_CODE_SHA=${CAMA_CODE_SHA}

USER cama
WORKDIR /home/cama
# The container stays up; experiments are driven with `docker exec`.
CMD ["sleep", "infinity"]
