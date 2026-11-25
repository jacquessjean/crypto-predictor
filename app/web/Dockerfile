# Build the React app, then serve the static files with unprivileged nginx.
FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json* ./
RUN npm install
COPY index.html vite.config.js ./
COPY src/ src/
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html