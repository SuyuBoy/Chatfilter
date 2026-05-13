import { useState, useEffect } from 'react';
import { EventList } from './components/EventList';
import { ClusterGrid } from './components/ClusterGrid';
import { ModelLoader } from './components/ModelLoader';
import { connectBridge, disconnectBridge } from './lib/bridge';
import { isModelReady } from '../app';

export function App() {
  const [modelDone, setModelDone] = useState(isModelReady());
  const [bridgeUrl, setBridgeUrl] = useState('ws://localhost:9696');
  const [token, setToken] = useState('');
  const [connected, setConnected] = useState(false);
  const [err, setErr] = useState('');

  useEffect(() => {
    return () => disconnectBridge();
  }, []);

  const handleConnect = () => {
    setErr('');
    try {
      connectBridge(bridgeUrl, token);
      setConnected(true);
    } catch (e: any) {
      setErr(e.message);
    }
  };

  const handleDisconnect = () => {
    disconnectBridge();
    setConnected(false);
  };

  if (!modelDone) {
    return <ModelLoader onReady={() => setModelDone(true)} />;
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top bar */}
      <div style={{
        background: 'var(--card)', borderBottom: '1px solid var(--border)',
        padding: '8px 16px', display: 'flex', alignItems: 'center', gap: 12,
        flexWrap: 'wrap',
      }}>
        <h1 style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent)', whiteSpace: 'nowrap' }}>
          弹幕语义聚类
        </h1>
        <span style={{ color: 'var(--border)' }}>|</span>
        <input
          value={bridgeUrl}
          onChange={e => setBridgeUrl(e.target.value)}
          placeholder="ws://host:9696"
          style={{
            width: 200, padding: '4px 8px', fontSize: 12, background: 'var(--bg)',
            color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 4, outline: 'none',
          }}
        />
        <input
          value={token}
          onChange={e => setToken(e.target.value)}
          placeholder="token (可选)"
          style={{
            width: 120, padding: '4px 8px', fontSize: 12, background: 'var(--bg)',
            color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 4, outline: 'none',
          }}
        />
        {!connected ? (
          <button onClick={handleConnect} style={{
            padding: '4px 16px', fontSize: 12, background: 'var(--accent)', color: '#fff',
            border: 'none', borderRadius: 4, cursor: 'pointer', fontWeight: 600,
          }}>
            连接
          </button>
        ) : (
          <button onClick={handleDisconnect} style={{
            padding: '4px 16px', fontSize: 12, background: 'var(--red)', color: '#fff',
            border: 'none', borderRadius: 4, cursor: 'pointer',
          }}>
            断开
          </button>
        )}
        {err && <span style={{ color: 'var(--red)', fontSize: 12 }}>{err}</span>}
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>
          BGE-small-zh-v1.5 | ONNX Runtime Web
        </span>
      </div>

      {/* Main content: two columns */}
      <div style={{
        flex: 1, display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 1,
        background: 'var(--border)', minHeight: 0, overflow: 'hidden',
      }}>
        <div style={{ overflow: 'hidden', minHeight: 0 }}>
          <EventList />
        </div>
        <div style={{ overflow: 'hidden', minHeight: 0 }}>
          <ClusterGrid />
        </div>
      </div>
    </div>
  );
}

export default App;
