FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python3 -c "import nltk; import ssl; \
    ssl._create_default_https_context = ssl._create_unverified_context; \
    nltk.download('punkt'); nltk.download('stopwords'); \
    nltk.download('wordnet'); nltk.download('omw-1.4'); \
    nltk.download('punkt_tab')"

# Copy backend application code
COPY backend/ .

# Copy ML code
COPY code/ /app/code/

# Create directory for ML models
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
