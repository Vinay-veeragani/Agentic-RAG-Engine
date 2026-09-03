# Maintained for CI/future use — not built or run in local dev on this machine
# (see docker-compose.yml header comment).

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY migrations ./migrations
COPY alembic.ini ./

EXPOSE 8000

CMD ["uvicorn", "agentic_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
