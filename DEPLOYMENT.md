# 🚀 CERT-EU Ticket Classification System - Deployment Guide

## Overview
This guide will help you deploy your CERT-EU Ticket Classification System to make it accessible to anyone via a web link.

## 🎯 Recommended Deployment Platform: Render

Render is perfect for this project because:
- ✅ Easy deployment with Docker
- ✅ Built-in PostgreSQL database
- ✅ Automatic HTTPS/SSL
- ✅ Custom domain support
- ✅ Environment variable management
- ✅ Free tier available

## 📋 Pre-Deployment Checklist

### 1. Prepare Your Repository
- [x] Docker configuration ready
- [x] Environment variables configured
- [x] Production startup script created
- [x] Frontend build process configured

### 2. Required Environment Variables
Set these in your deployment platform:
```
DATABASE_URL=postgresql://user:password@host:port/database
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ENVIRONMENT=production
```

## 🚀 Deployment Steps

### Option 1: Render (Recommended)

1. **Sign up for Render**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Connect Your Repository**
   - Click "New +" → "Web Service"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure Services**
   - Render will detect your Dockerfile
   - Add PostgreSQL database:
     - Click "New" → "Database" → "PostgreSQL"
   - Copy the DATABASE_URL from the database service

4. **Set Environment Variables**
   - Go to your service settings
   - Add all environment variables from the checklist above

5. **Deploy**
   - Render will automatically build and deploy
   - Your app will be available at `https://your-app-name.onrender.com`

### Option 2: Heroku

1. **Sign up for Render**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Create Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Choose "Docker" as the environment

3. **Configure Database**
   - Add PostgreSQL database
   - Copy the connection string

4. **Set Environment Variables**
   - Add all required environment variables

5. **Deploy**
   - Render will build and deploy automatically

### Option 3: Heroku

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Or download from heroku.com
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Add PostgreSQL**
   ```bash
   heroku addons:create heroku-postgresql:mini
   ```

4. **Set Environment Variables**
   ```bash
   heroku config:set SMTP_SERVER=smtp.gmail.com
   heroku config:set SMTP_PORT=587
   heroku config:set SMTP_USERNAME=your-email@gmail.com
   heroku config:set SMTP_PASSWORD=your-app-password
   heroku config:set ENVIRONMENT=production
   ```

5. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to production"
   git push heroku main
   ```

## 🔧 Post-Deployment Configuration

### 1. Test Your Deployment
- Visit your app URL
- Check that all pages load correctly
- Test the ticket analyzer
- Verify email reports work

### 2. Import Test Data (Optional)
If you want to populate with sample data:
```bash
# For Render (via their console)
python data_import.py

# For Heroku
heroku run python data_import.py
```

### 3. Set Up Custom Domain (Optional)
- Render: Go to settings → Domains
- Render: Go to settings → Custom Domains
- Heroku: Go to settings → Domains

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check DATABASE_URL format
   - Ensure database service is running

2. **Email Not Sending**
   - Verify SMTP credentials
   - Check if 2FA is enabled on Gmail
   - Use app-specific password

3. **Frontend Not Loading**
   - Check if build process completed
   - Verify nginx configuration

4. **Model Not Loading**
   - Check if model files are in the correct path
   - Verify file permissions

### Logs
- Render: View logs in the dashboard
- Render: Check logs in the service view
- Heroku: `heroku logs --tail`

## 📊 Monitoring

### Health Check
Your app includes a health check endpoint:
- `GET /api/health` - Returns system status

### Key Metrics to Monitor
- Response times
- Error rates
- Database connections
- Memory usage

## 🔒 Security Considerations

1. **Environment Variables**
   - Never commit sensitive data to git
   - Use strong passwords
   - Rotate credentials regularly

2. **Database Security**
   - Use connection pooling
   - Enable SSL connections
   - Regular backups

3. **API Security**
   - Rate limiting (consider adding)
   - Input validation
   - CORS configuration

## 🎉 Success!

Once deployed, your CERT-EU Ticket Classification System will be accessible to anyone with the URL. Share the link with your team or stakeholders to start using the system!

## 📞 Support

If you encounter issues:
1. Check the logs first
2. Verify environment variables
3. Test locally with Docker
4. Check platform-specific documentation

---

**Happy Deploying! 🚀**
