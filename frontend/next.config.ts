import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Keep the dev server's hot-reload files separate from production builds.
  distDir: process.env.NEXT_DIST_DIR || '.next',
};

export default nextConfig;
