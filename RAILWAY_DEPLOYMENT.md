# 🚀 Railway Deployment Guide for NBA 2K OCR System

This guide will help you deploy your NBA 2K OCR system to Railway.

## 📋 Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Railway CLI** (optional): Install with `npm i -g @railway/cli`

## 🚀 Deployment Steps

### Step 1: Connect to Railway

1. **Via Web Interface**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Via CLI** (optional):
   ```bash
   railway login
   railway init
   ```

### Step 2: Configure Environment Variables

In your Railway project dashboard, add these environment variables:

```bash
# Python Configuration
PYTHONPATH=/app

# Label Studio Configuration
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/data
LABEL_STUDIO_HOST=0.0.0.0
LABEL_STUDIO_PORT=8080

# Railway Configuration
RAILWAY_ENVIRONMENT=production
```

### Step 3: Configure Services

Railway will automatically detect your `docker-compose.yml` file. You can also use the Railway-specific compose file:

1. **Rename** `docker-compose.railway.yml` to `docker-compose.yml` for Railway deployment
2. **Or** keep both files and specify the Railway one in your deployment

### Step 4: Deploy

1. **Automatic Deployment**: Railway will automatically deploy when you push to your main branch
2. **Manual Deployment**: Use the Railway dashboard or CLI to trigger deployments

## 🔧 Configuration Files

### `railway.toml`
- Configures Railway-specific settings
- Defines health checks and restart policies
- Sets environment variables

### `Dockerfile`
- Main application container
- Installs dependencies and sets up the environment
- Exposes ports 8000 (dashboard) and 8080 (Label Studio)

### `docker-compose.railway.yml`
- Railway-optimized service configuration
- Includes health checks for both services
- Proper volume mounting for persistence

## 🌐 Accessing Your Application

After deployment, Railway will provide:

1. **Dashboard URL**: `https://your-app-name.railway.app` (port 8000)
2. **Label Studio URL**: `https://your-app-name.railway.app:8080`

## 📊 Monitoring

### Health Checks
- **Dashboard**: `/health` endpoint returns status
- **Label Studio**: `/api/status` endpoint for service status

### Logs
- View logs in Railway dashboard
- Use Railway CLI: `railway logs`

## 🔄 Updates

To update your deployment:

1. **Push changes** to your GitHub repository
2. **Railway automatically** redeploys
3. **Monitor** the deployment in the Railway dashboard

## 🛠️ Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check `requirements.txt` for missing dependencies
   - Verify Dockerfile syntax
   - Check `.dockerignore` excludes unnecessary files

2. **Service Not Starting**:
   - Check environment variables
   - Verify port configurations
   - Review health check endpoints

3. **Label Studio Issues**:
   - Ensure volumes are properly mounted
   - Check Label Studio configuration
   - Verify authentication tokens

### Debug Commands

```bash
# Check Railway status
railway status

# View logs
railway logs

# Connect to container
railway shell

# Check environment variables
railway variables
```

## 💰 Cost Optimization

1. **Use Railway's free tier** for development
2. **Scale down** when not in use
3. **Monitor usage** in Railway dashboard
4. **Consider** Railway's pay-as-you-go pricing

## 🔐 Security Considerations

1. **Environment Variables**: Store sensitive data in Railway variables
2. **Authentication**: Configure Label Studio authentication properly
3. **Access Control**: Use Railway's access controls
4. **HTTPS**: Railway provides automatic HTTPS

## 📈 Scaling

Railway supports automatic scaling:

1. **Horizontal Scaling**: Add more instances
2. **Vertical Scaling**: Increase resources
3. **Auto-scaling**: Configure based on traffic

## 🎯 Next Steps

After deployment:

1. **Test** all functionality
2. **Configure** Label Studio authentication
3. **Upload** sample images
4. **Monitor** performance
5. **Set up** alerts and notifications

## 📞 Support

- **Railway Documentation**: [docs.railway.app](https://docs.railway.app)
- **Railway Discord**: [discord.gg/railway](https://discord.gg/railway)
- **GitHub Issues**: Report bugs in your repository

---

**Happy Deploying! 🚀** 