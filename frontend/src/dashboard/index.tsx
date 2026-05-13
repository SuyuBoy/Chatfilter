import { createRoot } from 'react-dom/client';
import { App } from './App';

const el = document.getElementById('root')!;
el.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;min-height:100vh;color:var(--muted);font-family:sans-serif">Loading...</div>';

try {
  createRoot(el).render(<App />);
} catch (e: any) {
  el.innerHTML = `<div style="min-height:100vh;background:var(--bg);color:var(--red);padding:40px;font-family:monospace"><h2>Error</h2><pre>${e.message}\n${e.stack}</pre></div>`;
}
