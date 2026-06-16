# =====================================================================
# FILE: Dockerfile
# =====================================================================
FROM python:3.10-slim

# 1. Force Python to print logs instantly to catch hidden errors
ENV PYTHONUNBUFFERED=1

# 2. Install Linux image processing utilities (Requires Root)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Create non-root user (Required by Hugging Face Spaces security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH="${HOME}/.local/bin:${PATH}"

WORKDIR ${HOME}/app

# 4. Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all project files into the container
COPY --chown=user . .

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]