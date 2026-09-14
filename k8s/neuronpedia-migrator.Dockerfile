# Neuronpedia Migrator Image
# This image is used to run Prisma migrations before the webapp starts.
#
# Build this from the Neuronpedia webapp source directory:
#   docker build -t hitsai/neuronpedia-migrator:latest -f neuronpedia-migrator.Dockerfile .
#
# The webapp uses Next.js standalone output which doesn't include Prisma CLI.
# This image retains node_modules so `npx prisma` works.

FROM node:20-alpine

WORKDIR /app

# Install OpenSSL for Prisma
RUN apk add --no-cache openssl

# Copy package files
COPY package*.json ./
COPY prisma ./prisma/

# Install dependencies (including Prisma CLI)
RUN npm ci --only=production && \
    npm install prisma@5.22.0 @prisma/client@5.22.0

# Generate Prisma client
RUN npx prisma generate

# Default command (can be overridden in k8s)
CMD ["npx", "prisma", "db", "push", "--accept-data-loss", "--skip-generate"]
