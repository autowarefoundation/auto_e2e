import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Required by deploy/docker/Dockerfile.web (copies .next/standalone).
  output: "standalone",
  images: {
    // Camera thumbnails come from short-lived S3 presigned URLs; the Next.js
    // image optimizer cannot cache them, so they are always used unoptimized.
    unoptimized: true,
  },
};

export default nextConfig;
