#!/bin/bash

# CERT-EU Ticket Classification System Startup Script

echo "🚀 Starting CERT-EU Ticket Classification System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose and try again."
    exit 1
fi

echo "📦 Building and starting containers..."
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose exec postgres pg_isready -U cert_user -d cert_eu_db > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
fi

# Check Backend
if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ Backend API is ready"
else
    echo "⚠️  Backend API is not ready yet (this is normal if the model is still loading)"
fi

# Check Frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is ready"
else
    echo "❌ Frontend is not ready"
fi

echo ""
echo "🎉 CERT-EU Ticket Classification System is starting up!"
echo ""
echo "📍 Access URLs:"
echo "   Frontend:     http://localhost:3000"
echo "   Backend API:  http://localhost:8000"
echo "   API Docs:     http://localhost:8000/api/docs"
echo ""
echo "📊 To import test data:"
echo "   docker-compose exec backend python data_import.py"
echo ""
echo "🛑 To stop the system:"
echo "   docker-compose down"
echo ""
echo "📝 View logs:"
echo "   docker-compose logs -f"
echo ""

# Show container status
echo "📋 Container Status:"
docker-compose ps
