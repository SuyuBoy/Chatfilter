import { useClusterStore } from '../store/clusterStore';

function heat(ratio: number): string {
  const lr = ratio > 0 ? Math.log(1 + ratio * 9) / Math.log(10) : 0;
  return `rgba(${Math.round(60 + lr * 180)},${Math.round(120 - lr * 90)},${Math.round(250 - lr * 180)},0.25)`;
}

export const ClusterGrid: React.FC = () => {
  const { ingested, unique, clusters, permanent, cacheHitRate, connected, modelReady } = useClusterStore();
  const maxCount = Math.max(...clusters.map(c => c.count), 1);

  return (
    <div style={{
      background: 'var(--card)', borderRadius: 8, border: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden',
    }}>
      {/* Header stats */}
      <div style={{
        padding: '8px 12px', borderBottom: '1px solid var(--border)',
        display: 'flex', gap: 16, alignItems: 'center', fontSize: 12,
      }}>
        <span style={{ fontWeight: 600, color: 'var(--accent)' }}>语义聚类</span>
        <span>摄入 <b style={{ color: 'var(--accent)' }}>{ingested}</b></span>
        <span>唯一 <b style={{ color: 'var(--green)' }}>{unique}</b></span>
        <span>缓存 <b style={{ color: 'var(--purple)' }}>{(cacheHitRate * 100).toFixed(0)}%</b></span>
        <span style={{ flex: 1 }} />
        <span style={{ width: 8, height: 8, borderRadius: 4, display: 'inline-block',
          background: modelReady ? (connected ? 'var(--green)' : 'var(--yellow)') : 'var(--red)' }} />
        <span style={{ color: 'var(--muted)', fontSize: 11 }}>
          {!modelReady ? 'loading' : connected ? 'connected' : 'waiting'}
        </span>
      </div>

      {/* Permanent hotspots */}
      {permanent.length > 0 && (
        <div style={{ padding: '4px 12px', borderBottom: '1px solid var(--border)', display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 11, color: 'var(--purple)', fontWeight: 600 }}>热点:</span>
          {permanent.slice(0, 10).map(p => (
            <span key={p.id} style={{
              background: 'rgba(167,139,250,0.15)', color: 'var(--purple)',
              padding: '1px 6px', borderRadius: 3, fontSize: 11, whiteSpace: 'nowrap',
            }}>
              {p.canonical.slice(0, 8)} <b>{p.count}</b>
            </span>
          ))}
        </div>
      )}

      {/* Cluster grid */}
      <div style={{ flex: 1, overflow: 'auto', padding: 4 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 3 }}>
          {clusters.slice(0, 20).map((c, i) => (
            <div key={c.id} style={{
              background: heat(c.count / maxCount), borderRadius: 6, padding: '6px 8px',
              border: '1px solid rgba(255,255,255,0.06)', fontSize: 12,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, color: 'var(--accent)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  [{String(i + 1).padStart(2, '0')}] {c.canonical.slice(0, 10)}
                </span>
                <span style={{ color: 'var(--green)', flexShrink: 0, marginLeft: 4 }}>{c.count}次</span>
              </div>
              {c.examples[0] && (
                <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {c.examples[0].slice(0, 20)}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
