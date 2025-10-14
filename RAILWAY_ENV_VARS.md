# Railway Environment Variables

Add these environment variables in your Railway project settings:

## Required Environment Variables:

```
DATABASE_URL=sqlite:///./cert_eu.db
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SECRET_KEY=your-secret-key-here
ENVIRONMENT=production
```

## Optional Environment Variables:

```
PORT=8000
HOST=0.0.0.0
```

## Notes:
- Replace `your-email@gmail.com` with your actual Gmail
- Replace `your-app-password` with your Gmail App Password (not your regular password)
- Replace `your-secret-key-here` with a random secret key (you can generate one at https://randomkeygen.com)
- The DATABASE_URL will use SQLite by default, which is perfect for Railway
