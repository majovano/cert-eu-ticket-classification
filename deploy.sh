#!/bin/bash

# CERT-EU Ticket Classification System - Quick Deploy Script

echo "🚀 CERT-EU Ticket Classification System - Deployment Helper"
echo "=========================================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit for deployment"
fi

echo "🔍 Checking deployment readiness..."
echo ""

# Check if all required files exist
required_files=("Dockerfile" "railway.json" "start-production.sh" "backend/main.py" "frontend/package.json")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
    fi
done

echo ""
echo "🎯 Choose your deployment platform:"
echo "1) Railway (Recommended - Easiest)"
echo "2) Render (Good alternative)"
echo "3) Heroku (Classic choice)"
echo "4) Manual Docker deployment"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚂 Railway Deployment Instructions:"
        echo "1. Go to https://railway.app"
        echo "2. Sign up with GitHub"
        echo "3. Click 'New Project' → 'Deploy from GitHub repo'"
        echo "4. Select this repository"
        echo "5. Add PostgreSQL database service"
        echo "6. Set environment variables:"
        echo "   - DATABASE_URL (from PostgreSQL service)"
        echo "   - SMTP_SERVER=smtp.gmail.com"
        echo "   - SMTP_PORT=587"
        echo "   - SMTP_USERNAME=your-email@gmail.com"
        echo "   - SMTP_PASSWORD=your-app-password"
        echo "   - ENVIRONMENT=production"
        echo "7. Deploy!"
        echo ""
        echo "Your app will be available at: https://your-app-name.railway.app"
        ;;
    2)
        echo ""
        echo "🎨 Render Deployment Instructions:"
        echo "1. Go to https://render.com"
        echo "2. Sign up with GitHub"
        echo "3. Click 'New' → 'Web Service'"
        echo "4. Connect your GitHub repository"
        echo "5. Choose 'Docker' environment"
        echo "6. Add PostgreSQL database"
        echo "7. Set environment variables (same as Railway)"
        echo "8. Deploy!"
        echo ""
        echo "Your app will be available at: https://your-app-name.onrender.com"
        ;;
    3)
        echo ""
        echo "🟣 Heroku Deployment Instructions:"
        echo "1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli"
        echo "2. Run: heroku login"
        echo "3. Run: heroku create your-app-name"
        echo "4. Run: heroku addons:create heroku-postgresql:mini"
        echo "5. Set environment variables:"
        echo "   heroku config:set SMTP_SERVER=smtp.gmail.com"
        echo "   heroku config:set SMTP_PORT=587"
        echo "   heroku config:set SMTP_USERNAME=your-email@gmail.com"
        echo "   heroku config:set SMTP_PASSWORD=your-app-password"
        echo "   heroku config:set ENVIRONMENT=production"
        echo "6. Run: git push heroku main"
        echo ""
        echo "Your app will be available at: https://your-app-name.herokuapp.com"
        ;;
    4)
        echo ""
        echo "🐳 Manual Docker Deployment:"
        echo "1. Build the image: docker build -t cert-eu-app ."
        echo "2. Run with database: docker run -p 80:80 -e DATABASE_URL=your-db-url cert-eu-app"
        echo "3. Or use docker-compose: docker-compose up -d"
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "📚 For detailed instructions, see DEPLOYMENT.md"
echo "🎉 Happy deploying!"
