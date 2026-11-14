// Open the Help screen at a topic, from anywhere: remembers the topic, switches the panel through
// the UI store, and writes `#/help/<topic>` so the address can be bookmarked or shared.
import { helpHash, replaceHash } from '../../config/helpRoute';
import { useUIStore } from '../../stores/uiStore';
import { useHelpTopic } from './helpTopicStore';

export function openHelp(topic: string | null): void {
  useHelpTopic.getState().open(topic);
  useUIStore.getState().setActivePanel('help');
  replaceHash(helpHash(topic));
}
