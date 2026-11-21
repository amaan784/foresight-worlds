import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

// HTTPS so phones allow camera access; /api and /outputs go to the FastAPI backend.
// `FORESIGHT_HTTP=1 npm run dev` serves plain HTTP on :5174 (for a laptop-only browser on localhost).
const plainHttp = process.env.FORESIGHT_HTTP === "1" || process.env.npm_config_http === "true";

export default defineConfig({
  plugins: [react(), tailwindcss(), ...(plainHttp ? [] : [basicSsl()])],
  server: {
    host: "0.0.0.0",
    port: plainHttp ? 5174 : 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/outputs": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
