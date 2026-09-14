/**
 * Feature Groups WebSocket hook (Feature 010).
 *
 * Channel: extractions/{extraction_id}/feature-groups
 * Events:  feature_groups:progress | feature_groups:completed | feature_groups:failed
 */

import { useEffect, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useFeatureGroupsStore } from '../stores/featureGroupsStore';
import type { FeatureGroupsProgressEvent } from '../types/featureGroups';

export const useFeatureGroupsWebSocket = (extractionId: string | null) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { handleProgressEvent, handleCompletedEvent, handleFailedEvent, setWebSocketConnected } =
    useFeatureGroupsStore();
  const handlersRegisteredRef = useRef(false);

  useEffect(() => {
    setWebSocketConnected(isConnected);
  }, [isConnected, setWebSocketConnected]);

  useEffect(() => {
    if (handlersRegisteredRef.current) return;

    const handleProgress = (data: FeatureGroupsProgressEvent) => {
      handleProgressEvent(data.progress, data.stage);
    };
    const handleCompleted = () => handleCompletedEvent();
    const handleFailed = (data: { error: string }) => handleFailedEvent(data.error);

    on('feature_groups:progress', handleProgress);
    on('feature_groups:completed', handleCompleted);
    on('feature_groups:failed', handleFailed);
    handlersRegisteredRef.current = true;

    return () => {
      off('feature_groups:progress', handleProgress);
      off('feature_groups:completed', handleCompleted);
      off('feature_groups:failed', handleFailed);
      handlersRegisteredRef.current = false;
    };
  }, [on, off, handleProgressEvent, handleCompletedEvent, handleFailedEvent]);

  useEffect(() => {
    if (!extractionId || !isConnected) return;
    const channel = `extractions/${extractionId}/feature-groups`;
    subscribe(channel);
    return () => unsubscribe(channel);
  }, [extractionId, isConnected, subscribe, unsubscribe]);
};
