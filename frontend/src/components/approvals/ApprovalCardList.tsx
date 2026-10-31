/**
 * A list of waiting cards that moves focus to the next card after a successful decision (§2.7).
 */
import { useEffect, useRef, useState } from 'react';
import type { ApprovalRequest } from '../../types/featureGroups';
import { ApprovalCard, type ProbeRunContext } from './ApprovalCard';

export function ApprovalCardList({ requests, context }: { requests: ApprovalRequest[]; context: ProbeRunContext }) {
  const refs = useRef<Record<string, HTMLDivElement | null>>({});
  const [focusIndex, setFocusIndex] = useState<number | null>(null);

  useEffect(() => {
    if (focusIndex === null) return;
    const next = requests[Math.min(focusIndex, requests.length - 1)];
    if (next) refs.current[next.id]?.focus();
  }, [focusIndex, requests]);

  return (
    <div className="space-y-2">
      {requests.map((request, index) => (
        <ApprovalCard
          key={request.id}
          ref={(el) => {
            refs.current[request.id] = el;
          }}
          request={request}
          context={context}
          onDecided={() => setFocusIndex(index)}
        />
      ))}
    </div>
  );
}
