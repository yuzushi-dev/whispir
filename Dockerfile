FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY app.py .
COPY document_intelligence.py .

# Expose port 9999 for Streamlit
EXPOSE 9999

# Configure Streamlit behavior (disable browser telemetry, set port, etc.)
ENV STREAMLIT_SERVER_PORT=9999
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=2048

# Start the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py"]
