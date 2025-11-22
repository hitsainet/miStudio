import { useEffect, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';

export interface ActivationExample {
  prefix_tokens: string[];
  prime_token: string;
  suffix_tokens: string[];
  max_activation: number;
}

export interface LabelingResult {
  feature_id: number;
  label: string;
  category: string;
  description: string;
  examples: ActivationExample[];
}

export const useLabelingResultsWebSocket = (
  labelingJobId: string | null,
  onResult: (result: LabelingResult) => void
) => {
  const ws = useWebSocketContext();
  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;

  useEffect(() => {
    if (!labelingJobId) {
      console.log('[useLabelingResultsWebSocket] No labelingJobId provided');
      return;
    }

    console.log('[useLabelingResultsWebSocket] Subscribing to results for job:', labelingJobId);

    const channel = `labeling/${labelingJobId}/results`;

    const handleResult = (data: LabelingResult) => {
      console.log('[useLabelingResultsWebSocket] Received result:', data);
      onResultRef.current(data);
    };

    // Subscribe to the WebSocket channel (server-side subscription)
    console.log('[useLabelingResultsWebSocket] Subscribing to channel:', channel);
    ws.subscribe(channel);

    // Add event listener for the channel
    ws.on(channel, handleResult);

    return () => {
      console.log('[useLabelingResultsWebSocket] Unsubscribing from channel:', channel);
      ws.off(channel, handleResult);
      ws.unsubscribe(channel);
    };
  }, [labelingJobId, ws]);
};
