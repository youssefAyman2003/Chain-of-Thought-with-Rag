FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

# Copy application source
COPY . .

# Copy frontend static files
COPY frontend/ ./frontend/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--port", "8000"]