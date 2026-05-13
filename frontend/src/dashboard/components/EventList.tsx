import { useEffect, useRef } from 'react';
import { useChatStore } from '../store/chatStore';

export const EventList: React.FC = () => {
  const messages = useChatStore(s => s.messages);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) ref.current.scrollTo({ top: ref.current.scrollHeight, behavior: 'smooth' });
  }, [messages.length]);

  return (
    <div style={{
      background: 'var(--card)', borderRadius: 8, border: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden',
    }}>
      <div style={{
        padding: '8px 12px', borderBottom: '1px solid var(--border)',
        fontSize: 13, fontWeight: 600, color: 'var(--accent)',
      }}>
        弹幕流 ({messages.length})
      </div>
      <div ref={ref} style={{ flex: 1, overflow: 'auto', padding: '8px 12px' }}>
        {messages.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--muted)', padding: 32, fontSize: 13 }}>
            等待弹幕...
          </div>
        ) : (
          messages.map((e: any, i) => (
            <div key={e.id || i} style={{ padding: '4px 0', borderBottom: '1px solid rgba(255,255,255,0.04)', fontSize: 12 }}>
              {e.type === 'message' && (
                <div style={{ display: 'flex', gap: 8 }}>
                  <span style={{ color: 'var(--accent)', fontWeight: 600, flexShrink: 0 }}>
                    {e.username || '??'}:
                  </span>
                  <span style={{ wordBreak: 'break-all' }}>{e.message}</span>
                </div>
              )}
              {e.type === 'superchat' && (
                <div style={{ color: 'var(--yellow)' }}>
                  <b>{e.username}</b> [¥{e.priceNormalized}] {e.message}
                </div>
              )}
              {e.type === 'gift' && (
                <div style={{ color: '#f472b6' }}>
                  {e.username}: {e.giftName} ×{e.num}
                </div>
              )}
              {e.type === 'interaction' && (
                <div style={{ color: 'var(--muted)', fontSize: 11 }}>
                  {e.username} 进入直播间
                </div>
              )}
              {e.type === 'entry-effect' && (
                <div style={{ color: '#38bdf8' }}>{e.message}</div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
