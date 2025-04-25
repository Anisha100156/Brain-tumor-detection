import path from 'path';

const nextConfig = {
  webpack(config) {
    config.resolve.alias = {
      '@': path.resolve('my-app'), // Update the alias to point to the 'my-app' folder
    };
    return config;
  },
};

export default nextConfig;
