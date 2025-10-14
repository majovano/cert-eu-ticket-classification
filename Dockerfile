# Multi-stage build for production deployment
FROM node:18-alpine AS frontend-builder

# Build frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install --omit=dev
COPY frontend/ ./
RUN npm run build

# Python backend stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .
COPY code/ /app/code/

# Copy built frontend
COPY --from=frontend-builder /app/frontend/build /app/frontend/build

# Create nginx config
RUN echo 'server { \
    listen 80; \
    server_name _; \
    \
    location /api { \
        proxy_pass http://localhost:8000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
    } \
    \
    location / { \
        root /app/frontend/build; \
        index index.html; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/sites-available/default

# Create startup script
RUN echo '#!/bin/bash\n\
# Start backend in background\n\
uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
\n\
# Start nginx in foreground\n\
nginx -g "daemon off;"' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 80

# Start the application
CMD ["/app/start.sh"]
