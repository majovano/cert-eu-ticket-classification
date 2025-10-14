#!/bin/bash

# Production startup script for Railway deployment

echo "🚀 Starting CERT-EU Ticket Classification System in Production..."

# Set environment variables for production
export DATABASE_URL=${DATABASE_URL:-"sqlite:///./cert_eu.db"}
export PYTHONPATH="/app:/app/code"

# Create database if it doesn't exist
python -c "
import sqlite3
import os
db_path = '/app/cert_eu.db'
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    conn.close()
    print('✅ Database created')
else:
    print('✅ Database exists')
"

# Start the application
echo "🌐 Starting web server..."
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
