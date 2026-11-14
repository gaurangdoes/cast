import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// base './' so the build works when served from web/dist by the Python app.
export default defineConfig({
  base: './',
  plugins: [react()],
  server: { port: 5173 },
})
