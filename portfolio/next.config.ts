/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export", // if you’re exporting a static site
  eslint: {
    // This tells Next.js to skip ESLint errors at build time
    ignoreDuringBuilds: true,
  },
  // …any other settings…
};

module.exports = nextConfig;
