/** @type {import('next').NextConfig} */
const path = require('path');

const nextConfig = {
  webpack(config) {
    config.resolve.alias = {
      '@': path.resolve(__dirname, 'my-app'), // Update the alias to point to the 'my-app' folder
    };
    return config;
  },
};

export default nextConfig;
