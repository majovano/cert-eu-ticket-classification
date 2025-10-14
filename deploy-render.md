# Deploy to Render.com

## Quick Deployment Steps:

1. **Go to**: https://render.com
2. **Sign up/Login** with GitHub
3. **Click**: "New +" → "Web Service"
4. **Connect Repository**: Select `majovano/cert-eu-ticket-classification`
5. **Configure**:
   - **Name**: `cert-eu-ticket-classification`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: `Free`
6. **Add Environment Variables**:
   - `DATABASE_URL`: `sqlite:///./cert_eu.db`
   - `SECRET_KEY`: (auto-generated)
   - `SMTP_SERVER`: `smtp.gmail.com`
   - `SMTP_PORT`: `587`
   - `SMTP_USERNAME`: `mjovanovjr@gmail.com`
   - `SMTP_PASSWORD`: `ttizkzzoregfazqy`
7. **Click**: "Create Web Service"
8. **Wait**: 5-10 minutes for build and deployment
9. **Access**: Your app will be available at `https://cert-eu-ticket-classification.onrender.com`

## That's it! 🚀

Render will automatically:
- Build your Docker container
- Deploy both frontend and backend
- Give you a public URL
- Auto-deploy on every GitHub push
