import { Sun, Moon } from 'lucide-react';
import { CompactGPUStatus } from '../SystemMonitor/CompactGPUStatus';
import { COMPONENTS } from '../../config/brand';
import { useUIStore } from '../../stores/uiStore';

interface HeaderProps {
  theme: 'light' | 'dark';
  onThemeToggle: () => void;
  onNavigateToMonitor: () => void;
}

export function Header({ theme, onThemeToggle, onNavigateToMonitor }: HeaderProps) {
  const { sidebar } = useUIStore();
  const { collapsed } = sidebar;

  return (
    <header
      className={`
        sticky top-0 z-30 h-14
        bg-white dark:bg-slate-950
        border-b border-slate-200 dark:border-slate-800
        transition-all duration-300 ease-in-out
        ${collapsed ? 'ml-16' : 'ml-56'}
      `}
    >
      <div className="h-full px-6 flex items-center justify-end gap-4">
        <CompactGPUStatus onClickMonitor={onNavigateToMonitor} />
        <button
          onClick={onThemeToggle}
          className={COMPONENTS.button.icon}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>
      </div>
    </header>
  );
}
