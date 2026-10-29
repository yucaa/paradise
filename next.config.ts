import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The piece is a static page in public/live/.
  async redirects() {
    return [{ source: "/", destination: "/live/index.html", permanent: false }];
  },
};

export default nextConfig;
