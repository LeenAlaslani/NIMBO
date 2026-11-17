import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backend = process.env.NIMBO_API_TARGET || 'http://127.0.0.1:8005'
const proxy = { '^/api(?:/|$)': backend, '^/nimbo(?:/|$)': backend }

export default defineConfig({
  plugins: [react()],
  server: { proxy },
  preview: { host: '127.0.0.1', proxy },
})
