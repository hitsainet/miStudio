# Frontend Dockerfile for miStudio
# Multi-stage build supporting development and production

# ==============================================================================
# Base stage - shared dependencies
# ==============================================================================
FROM node:20-alpine AS base

WORKDIR /app

# Install dependencies only (for better caching)
COPY package.json package-lock.json* ./

# ==============================================================================
# Development stage - Vite dev server with HMR
# ==============================================================================
FROM base AS development

# Install all dependencies including devDependencies
RUN npm ci

# Copy source code
COPY . .

# Expose Vite dev server port
EXPOSE 3000

# Environment variables for development
ENV NODE_ENV=development \
    VITE_API_URL= \
    VITE_WS_URL=

# Start Vite dev server
# --host 0.0.0.0 allows access from outside container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# ==============================================================================
# Build stage - compile TypeScript and build production assets
# ==============================================================================
FROM base AS builder

# Install all dependencies (need devDependencies for build)
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# ==============================================================================
# Production stage - serve static files with nginx
# ==============================================================================
FROM nginx:alpine AS production

# Copy custom nginx config for SPA routing
RUN echo 'server { \n\
    listen 80; \n\
    server_name localhost; \n\
    root /usr/share/nginx/html; \n\
    index index.html; \n\
    \n\
    # Gzip compression \n\
    gzip on; \n\
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript; \n\
    \n\
    # Cache static assets \n\
    location /assets/ { \n\
        expires 1y; \n\
        add_header Cache-Control "public, immutable"; \n\
    } \n\
    \n\
    # SPA fallback - serve index.html for all routes \n\
    location / { \n\
        try_files $uri $uri/ /index.html; \n\
    } \n\
    \n\
    # Health check endpoint \n\
    location /health { \n\
        access_log off; \n\
        return 200 "healthy\\n"; \n\
        add_header Content-Type text/plain; \n\
    } \n\
}' > /etc/nginx/conf.d/default.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
