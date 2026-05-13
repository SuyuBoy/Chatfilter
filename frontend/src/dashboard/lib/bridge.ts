import { LaplaceEventBridgeClient } from '@laplace.live/event-bridge-sdk';
import type { LaplaceEvent } from '@laplace.live/event-types';
import { useChatStore } from '../store/chatStore';
import { useClusterStore } from '../store/clusterStore';

let client: LaplaceEventBridgeClient | null = null;

export function connectBridge(url: string, token = '') {
  disconnectBridge();

  client = new LaplaceEventBridgeClient({ url, token, reconnect: true });
  const { addMessage } = useChatStore.getState();
  const { ingest, setConnected } = useClusterStore.getState();

  client.onConnectionStateChange(state => {
    setConnected(state === 'connected');
  });

  client.onAny((event: LaplaceEvent) => {
    addMessage(event);
    if (event.type === 'message' && (event as any).message) {
      ingest((event as any).message, (event as any).username || '');
    }
  });

  client.connect().catch(err => {
    console.error('Bridge connect failed:', err);
    setConnected(false);
  });
}

export function disconnectBridge() {
  client?.disconnect();
  client = null;
}

export function getBridgeClient() { return client; }
