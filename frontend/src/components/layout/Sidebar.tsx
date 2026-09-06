import {
  Database,
  Server,
  GraduationCap,
  Layers,
  Tags,
  Network,
  Sliders,
  FileText,
  Activity,
  Settings,
  ChevronLeft,
  ChevronRight,
  BookOpen,
  Boxes,
  GitBranch,
  Sparkles,
} from 'lucide-react';
import logoSvg from '../../assets/logo.svg';
import { useUIStore } from '../../stores/uiStore';
import type { ActivePanel } from '../../config/panels';
import { BRAND } from '../../config/brand';

const APP_VERSION = '0.5.0';

/**
 * Nav order IS array order — there is no sort key anywhere. Exported so tests
 * assert placement against the real registry rather than against a copy of it.
 */
export const navItems: { id: ActivePanel; label: string; icon: typeof Database }[] = [
  { id: 'models', label: 'Models', icon: Server },
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'saes', label: 'SAEs', icon: Network },
  { id: 'templates', label: 'Templates', icon: FileText },
  { id: 'training', label: 'Training', icon: GraduationCap },
  { id: 'extractions', label: 'Extractions', icon: Layers },
  { id: 'labeling', label: 'Labeling', icon: Tags },
  { id: 'feature-groups', label: 'Clusters', icon: Boxes },
  { id: 'circuits', label: 'Circuits', icon: GitBranch },
  // Ordering here IS the nav order — there is no sort key. J-Lens sits
  // immediately before Steering (FPRD §3.1); moving this line moves the tab.
  { id: 'jlens', label: 'J-Lens', icon: Sparkles },
  { id: 'steering', label: 'Steering', icon: Sliders },
  { id: 'system', label: 'Monitor', icon: Activity },
];

export const bottomNavItems: { id: ActivePanel; label: string; icon: typeof Database }[] = [
  { id: 'settings', label: 'Settings', icon: Settings },
];

interface SidebarProps {
  activePanel: ActivePanel;
  onPanelChange: (panel: ActivePanel) => void;
}

export function Sidebar({ activePanel, onPanelChange }: SidebarProps) {
  const { sidebar, toggleSidebar } = useUIStore();
  const { collapsed } = sidebar;

  return (
    <aside
      className={`
        fixed left-0 top-0 h-screen
        bg-white dark:bg-slate-900/95 border-r border-slate-200 dark:border-slate-700/50
        backdrop-blur-sm z-40
        transition-all duration-300 ease-in-out
        ${collapsed ? 'w-16' : 'w-56'}
      `}
    >
      {/* Logo */}
      <div className="h-14 flex items-center px-4 border-b border-slate-200 dark:border-slate-700/50">
        <div className="flex items-center gap-2 min-w-0">
          <img
            src={logoSvg}
            alt={BRAND.name}
            className="w-8 h-8 flex-shrink-0"
          />
          {!collapsed && (
            <div className="overflow-hidden min-w-0 flex-1">
              <div className="text-sm font-semibold text-slate-900 dark:text-slate-100 truncate">{BRAND.name}</div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight truncate">
                {BRAND.tagline}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Navigation - flex column to push settings to bottom */}
      <div className="flex flex-col h-[calc(100vh-3.5rem)]">
        <nav className="p-2 space-y-0.5 flex-1">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => onPanelChange(item.id)}
              className={`
                w-full flex items-center gap-3 px-3 py-2.5 rounded-lg
                transition-all duration-200
                ${activePanel === item.id
                  ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'
                }
                ${collapsed ? 'justify-center' : ''}
              `}
              title={collapsed ? item.label : undefined}
            >
              <item.icon className="w-5 h-5 flex-shrink-0" />
              {!collapsed && (
                <span className="text-sm font-medium">{item.label}</span>
              )}
            </button>
          ))}
        </nav>

        {/* Bottom nav (Settings + Manual link) + version badge */}
        <div className="p-2 border-t border-slate-200 dark:border-slate-700/50 space-y-1">
          {bottomNavItems.map((item) => (
            <div key={item.id} className="flex items-center">
              <button
                onClick={() => onPanelChange(item.id)}
                className={`
                  flex-1 flex items-center gap-3 px-3 py-2.5 rounded-lg
                  transition-all duration-200
                  ${activePanel === item.id
                    ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'
                  }
                  ${collapsed ? 'justify-center' : ''}
                `}
                title={collapsed ? item.label : undefined}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                {!collapsed && (
                  <span className="text-sm font-medium">{item.label}</span>
                )}
              </button>
              {!collapsed && (
                <a
                  href="https://hitsainet.github.io/miStudio/"
                  target="_blank"
                  rel="noopener noreferrer"
                  title="User Manual"
                  className="flex-shrink-0 p-2 rounded hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                >
                  <BookOpen className="w-4 h-4 text-slate-600 dark:text-slate-400 hover:text-emerald-400" />
                </a>
              )}
            </div>
          ))}
          {!collapsed && (
            <div className="px-3 pt-1">
              <a
                href="https://github.com/hitsainet/miStudio/releases"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[10px] text-slate-600 dark:text-slate-400 hover:text-emerald-400 transition-colors font-mono"
              >
                v{APP_VERSION}
              </a>
            </div>
          )}
        </div>
      </div>

      {/* Collapse Toggle */}
      <button
        onClick={toggleSidebar}
        className="absolute -right-3 top-20 w-6 h-6 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
      >
        {collapsed ? (
          <ChevronRight className="w-3 h-3" />
        ) : (
          <ChevronLeft className="w-3 h-3" />
        )}
      </button>
    </aside>
  );
}
