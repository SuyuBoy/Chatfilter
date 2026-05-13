import { useState, useEffect } from 'react';
import { initAuto, isModelReady } from '../../app';
import { useClusterStore } from '../store/clusterStore';

export const ModelLoader: React.FC<{ onReady: () => void }> = ({ onReady }) => {
  const [progress, setProgress] = useState('');
  const [pct, setPct] = useState(0);
  const [err, setErr] = useState('');
  const setModelReady = useClusterStore(s => s.setModelReady);

  useEffect(() => {
    if (isModelReady()) { onReady(); return; }
    (async () => {
      try {
        await initAuto(msg => {
          setProgress(msg);
          // Parse download progress like "Downloading onnx/model.onnx..."
          const m = msg.match(/(\d+)%/);
          if (m) setPct(parseInt(m[1]));
          else setPct(p => Math.min(p + 5, 95));
        });
        setPct(100);
        setProgress('Model ready');
        setModelReady(true);
        setTimeout(onReady, 500);
      } catch (e: any) {
        setErr(e.message || String(e));
      }
    })();
  }, []);

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100,
    }}>
      <div style={{
        background: 'var(--card)', borderRadius: 12, padding: 32, maxWidth: 420, width: '90%',
        textAlign: 'center', border: '1px solid var(--border)',
      }}>
        <h2 style={{ marginBottom: 8, color: 'var(--accent)' }}>Loading Model</h2>
        <p style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 24 }}>
          BGE-small-zh-v1.5 (~90MB)
        </p>

        {/* Progress bar */}
        <div style={{
          height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden', marginBottom: 12,
        }}>
          <div style={{
            height: '100%', width: `${pct}%`, background: 'linear-gradient(90deg, var(--accent), var(--purple))',
            borderRadius: 4, transition: 'width 0.3s',
          }} />
        </div>

        <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>{pct}%</p>
        {progress && <p style={{ fontSize: 12, color: 'var(--muted)' }}>{progress}</p>}
        {err && <p style={{ fontSize: 12, color: 'var(--red)', marginTop: 12 }}>{err}</p>}
      </div>
    </div>
  );
};
