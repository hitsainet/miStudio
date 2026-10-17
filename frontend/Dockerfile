# Frontend Dockerfile for miStudio
# Multi-stage build supporting development and production

# ==============================================================================
# Base stage - shared dependencies
# ==============================================================================
FROM node:22-alpine AS base

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
# Production stage - serve static files with nginx (non-root)
# ==============================================================================
# nginxinc/nginx-unprivileged:alpine runs as the 'nginx' user (uid 101) by
# default and listens on port 8080 rather than 80 so it doesn't need
# CAP_NET_BIND_SERVICE. This satisfies Docker Scout's "default non-root user"
# policy and is the recommended way to run nginx in container orchestrators.
FROM nginxinc/nginx-unprivileged:alpine AS production

# Custom nginx config for SPA routing. Listens on 8080 (the default for the
# unprivileged image). Downstream Services/Ingresses map external port 80 →
# container port 8080.
COPY --chown=nginx:nginx <<'EOF' /etc/nginx/conf.d/default.conf
server {
    listen 8080;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Copy built assets from builder stage
COPY --from=builder --chown=nginx:nginx /app/dist /usr/share/nginx/html

# Copy LICENSE file for open source compliance
COPY --chown=nginx:nginx LICENSE /usr/share/nginx/html/LICENSE

# Expose the unprivileged port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:8080/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
