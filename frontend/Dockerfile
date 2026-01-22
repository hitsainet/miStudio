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

# Update Alpine packages to fix CVEs (libpng vulnerabilities)
RUN apk update && apk upgrade --no-cache libpng

# Copy custom nginx config for SPA routing
# Listen on both port 80 and 3000 for compatibility with docker nginx proxy
RUN printf '%s\n' \
    'server {' \
    '    listen 80;' \
    '    listen 3000;' \
    '    server_name localhost;' \
    '    root /usr/share/nginx/html;' \
    '    index index.html;' \
    '' \
    '    gzip on;' \
    '    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;' \
    '' \
    '    location /assets/ {' \
    '        expires 1y;' \
    '        add_header Cache-Control "public, immutable";' \
    '    }' \
    '' \
    '    location / {' \
    '        try_files $uri $uri/ /index.html;' \
    '    }' \
    '' \
    '    location /health {' \
    '        access_log off;' \
    '        return 200 "healthy\n";' \
    '        add_header Content-Type text/plain;' \
    '    }' \
    '}' > /etc/nginx/conf.d/default.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy LICENSE file for open source compliance
COPY LICENSE /usr/share/nginx/html/LICENSE

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
