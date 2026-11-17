# Multi-stage build: small, non-root runtime image for the agent API.
FROM python:3.12-slim AS build
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[graph]"

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=build /install /usr/local
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
# Secrets (OPENAI_API_KEY, ...) come from the environment at runtime — never bake them into the image.
CMD ["uvicorn", "ailearn.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
