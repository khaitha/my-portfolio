/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  eslint: {
    ignoreDuringBuilds: true, // skip ESLint errors at build time
  },
  // …any other config you had…
};

module.exports = nextConfig;
