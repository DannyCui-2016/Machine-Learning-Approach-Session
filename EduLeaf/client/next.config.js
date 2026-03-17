/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001',
    NEXT_PUBLIC_ML_URL: process.env.NEXT_PUBLIC_ML_URL || 'http://localhost:8000',
  },
}

module.exports = nextConfig
