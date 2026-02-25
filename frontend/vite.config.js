import { createLogger, defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';
import fs from 'node:fs';

// Silence expected proxy errors so the console stays clean.
// ECONNRESET  = backend restarted while Vite held an open WS connection (normal)
// ECONNREFUSED = backend not running yet (normal on first start)
// ECONNABORTED = browser aborted request / page reload during WS writes (normal)
const SILENT_CODES = new Set(['ECONNRESET', 'ECONNREFUSED', 'ENOTFOUND', 'ECONNABORTED']);

function makeProxyHandler(label) {
  return (proxy) => {
    const filteredErrorHandler = (err, _req, res) => {
      if (SILENT_CODES.has(err?.code)) return; // suppress expected disconnects
      const code = err?.code ? ` (${err.code})` : '';
      console.warn(`[${label} proxy]${code}`, err?.message ?? String(err));
      // If it's an HTTP request (not WS) still send a response so the page doesn't hang
      if (res && typeof res.writeHead === 'function' && !res.headersSent) {
        res.writeHead(502, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: false, error: 'backend unavailable' }));
      }
    };

    const installFilteredHandler = () => {
      for (const listener of proxy.listeners('error')) {
        if (listener !== filteredErrorHandler) proxy.off('error', listener);
      }
      if (!proxy.listeners('error').includes(filteredErrorHandler)) {
        proxy.on('error', filteredErrorHandler);
      }
    };

    // Vite has changed the timing of when it attaches its own error handler across
    // versions. Install ours now, and also re-install if a new handler appears later.
    installFilteredHandler();
    setTimeout(installFilteredHandler, 0);
    setImmediate(installFilteredHandler);
    proxy.on('newListener', (event, listener) => {
      if (event !== 'error') return;
      if (listener === filteredErrorHandler) return;
      setImmediate(installFilteredHandler);
    });
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  const silentWsProxyCodes = new Set(['ECONNRESET', 'ECONNREFUSED', 'ECONNABORTED', 'ENOTFOUND']);
  const baseLogger = createLogger();
  const customLogger = {
    ...baseLogger,
    error(msg, options) {
      const err = options?.error;
      const code = err?.code || err?.cause?.code;
      if (
        typeof msg === 'string' &&
        (msg.includes('ws proxy socket error:') || msg.includes('ws proxy error:')) &&
        silentWsProxyCodes.has(code)
      ) {
        return;
      }
      return baseLogger.error(msg, options);
    },
  };

  const voiceTarget = env.VOICE_PROXY_TARGET || 'http://127.0.0.1:8010';
  // Default to HTTPS so mobile can use mic permissions (secure context).
  // Override with VITE_HTTPS=false if you explicitly need plain HTTP.
  const useHttps = String(env.VITE_HTTPS ?? 'true').toLowerCase() === 'true';

  const sslKeyPath = env.VITE_SSL_KEY_PATH || '';
  const sslCertPath = env.VITE_SSL_CERT_PATH || '';
  const hasCustomSsl =
    useHttps && sslKeyPath && sslCertPath && fs.existsSync(sslKeyPath) && fs.existsSync(sslCertPath);
  const httpsConfig = hasCustomSsl
    ? { key: fs.readFileSync(sslKeyPath), cert: fs.readFileSync(sslCertPath) }
    : useHttps;

  return {
    customLogger,
    plugins: useHttps ? (hasCustomSsl ? [react()] : [react(), basicSsl()]) : [react()],
    server: {
      host: true,
      https: httpsConfig,
      hmr: useHttps ? { protocol: 'wss' } : undefined,
      proxy: {
        '/voice': {
          target: voiceTarget,
          changeOrigin: true,
          ws: true,
          rewrite: (path) => path.replace(/^\/voice/, ''),
          configure: makeProxyHandler('voice'),
        },
        // Back-compat with older frontend configs that hit `/status` directly.
        '/status': {
          target: voiceTarget,
          changeOrigin: true,
          configure: makeProxyHandler('voice-status'),
        },
      },
    },
  };
});
