/**
 * Brand Constants and Design System Configuration
 *
 * Centralized brand identity configuration for MechInterp Studio.
 * This file defines colors, typography, spacing, and component variants
 * to ensure consistent brand application across the application.
 */

/**
 * Core brand identity
 */
export const BRAND = {
  name: 'MechInterp Studio',
  shortName: 'miStudio',
  tagline: 'Edge AI Feature Discovery Platform',
  description: 'Professional tool for training and analyzing Sparse Autoencoders',
  version: '0.1.0',
  repository: 'https://github.com/Onegaishimas/miStudio',
} as const;

/**
 * Brand color palette
 * Based on Tailwind CSS color system with semantic naming
 */
export const COLORS = {
  // Primary brand color (emerald)
  brand: {
    primary: '#10b981',    // emerald-500
    hover: '#059669',      // emerald-600
    light: '#34d399',      // emerald-400
    dark: '#047857',       // emerald-700
  },

  // Surface colors (slate dark theme)
  surface: {
    base: '#0f1629',       // slate-950 (custom)
    elevated: '#1e293b',   // slate-900
    card: '#334155',       // slate-800
    overlay: '#475569',    // slate-700
  },

  // Text colors
  text: {
    primary: '#f1f5f9',    // slate-100
    secondary: '#cbd5e1',  // slate-300
    muted: '#94a3b8',      // slate-400
    disabled: '#64748b',   // slate-500
  },

  // Border colors
  border: {
    default: '#334155',    // slate-800
    muted: '#475569',      // slate-700
    focus: '#10b981',      // emerald-500
  },

  // Status colors
  status: {
    success: '#10b981',    // emerald-500
    warning: '#f59e0b',    // amber-500
    error: '#ef4444',      // red-500
    info: '#3b82f6',       // blue-500
    processing: '#8b5cf6', // violet-500
  },

  // Dataset/Model status badges
  badge: {
    downloading: {
      bg: 'bg-blue-500/20',
      text: 'text-blue-400',
      border: 'border-blue-500/30',
    },
    processing: {
      bg: 'bg-yellow-500/20',
      text: 'text-yellow-400',
      border: 'border-yellow-500/30',
    },
    ready: {
      bg: 'bg-emerald-500/20',
      text: 'text-emerald-400',
      border: 'border-emerald-500/30',
    },
    error: {
      bg: 'bg-red-500/20',
      text: 'text-red-400',
      border: 'border-red-500/30',
    },
  },
} as const;

/**
 * Typography system
 */
export const TYPOGRAPHY = {
  // Font families
  fonts: {
    heading: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
    body: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
    mono: '"SF Mono", Monaco, "Cascadia Code", Consolas, "Courier New", monospace',
  },

  // Font sizes (matching Tailwind scale)
  sizes: {
    xs: '0.75rem',      // 12px - badges, labels
    sm: '0.875rem',     // 14px - secondary text
    base: '1rem',       // 16px - body text
    lg: '1.125rem',     // 18px - section headers
    xl: '1.25rem',      // 20px - panel titles
    '2xl': '1.5rem',    // 24px - page headers
    '3xl': '1.875rem',  // 30px - large headings
  },

  // Font weights
  weights: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },

  // Line heights
  lineHeights: {
    tight: 1.25,
    normal: 1.5,
    relaxed: 1.75,
  },
} as const;

/**
 * Spacing system (based on Tailwind's 4px base unit)
 */
export const SPACING = {
  0: '0',
  1: '0.25rem',   // 4px
  2: '0.5rem',    // 8px
  3: '0.75rem',   // 12px
  4: '1rem',      // 16px
  6: '1.5rem',    // 24px
  8: '2rem',      // 32px
  12: '3rem',     // 48px
  16: '4rem',     // 64px
  24: '6rem',     // 96px
} as const;

/**
 * Border radius values
 */
export const RADIUS = {
  none: '0',
  sm: '0.125rem',   // 2px
  default: '0.25rem', // 4px
  md: '0.375rem',   // 6px
  lg: '0.5rem',     // 8px
  xl: '0.75rem',    // 12px
  full: '9999px',
} as const;

/**
 * Component style presets
 * Reusable className strings for consistent component styling
 */
export const COMPONENTS = {
  // Card variants
  card: {
    base: 'bg-slate-900/50 dark:bg-white/90 border border-slate-800 dark:border-slate-200 rounded-lg',
    elevated: 'bg-slate-900 dark:bg-white border border-slate-700 dark:border-slate-300 rounded-lg shadow-lg',
    interactive: 'bg-slate-900/50 dark:bg-white/90 border border-slate-800 dark:border-slate-200 rounded-lg hover:border-slate-700 dark:hover:border-slate-400 transition-colors cursor-pointer',
  },

  // Button variants
  button: {
    primary: 'px-4 py-2 bg-emerald-600 hover:bg-emerald-700 dark:bg-emerald-500 dark:hover:bg-emerald-600 disabled:bg-slate-700 dark:disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-medium rounded transition-colors',
    secondary: 'px-4 py-2 bg-slate-800 hover:bg-slate-700 dark:bg-slate-100 dark:hover:bg-slate-200 disabled:bg-slate-900 dark:disabled:bg-slate-50 disabled:cursor-not-allowed text-slate-300 dark:text-slate-700 disabled:text-slate-600 dark:disabled:text-slate-400 font-medium rounded transition-colors',
    ghost: 'px-4 py-2 hover:bg-slate-800 dark:hover:bg-slate-100 text-slate-400 dark:text-slate-600 hover:text-slate-300 dark:hover:text-slate-700 rounded transition-colors',
    danger: 'px-4 py-2 bg-red-600 hover:bg-red-700 dark:bg-red-500 dark:hover:bg-red-600 disabled:bg-slate-700 dark:disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-medium rounded transition-colors',
    icon: 'p-2 hover:bg-slate-800 dark:hover:bg-slate-100 text-slate-400 dark:text-slate-600 hover:text-slate-300 dark:hover:text-slate-700 rounded transition-colors',
  },

  // Input variants
  input: {
    default: 'w-full px-4 py-2 bg-slate-800 dark:bg-white border border-slate-700 dark:border-slate-300 rounded text-slate-100 dark:text-slate-900 placeholder-slate-500 dark:placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent',
    error: 'w-full px-4 py-2 bg-slate-800 dark:bg-white border border-red-500 rounded text-slate-100 dark:text-slate-900 placeholder-slate-500 dark:placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent',
    textarea: 'w-full px-4 py-2 bg-slate-800 dark:bg-white border border-slate-700 dark:border-slate-300 rounded text-slate-100 dark:text-slate-900 placeholder-slate-500 dark:placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none',
  },

  // Badge variants (for status indicators)
  badge: {
    default: 'inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border',
    success: 'inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border bg-emerald-500/20 dark:bg-emerald-100 text-emerald-400 dark:text-emerald-700 border-emerald-500/30 dark:border-emerald-300',
    warning: 'inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border bg-yellow-500/20 dark:bg-yellow-100 text-yellow-400 dark:text-yellow-700 border-yellow-500/30 dark:border-yellow-300',
    error: 'inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border bg-red-500/20 dark:bg-red-100 text-red-400 dark:text-red-700 border-red-500/30 dark:border-red-300',
    info: 'inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border bg-blue-500/20 dark:bg-blue-100 text-blue-400 dark:text-blue-700 border-blue-500/30 dark:border-blue-300',
  },

  // Progress bar
  progress: {
    container: 'w-full h-2 bg-slate-800 dark:bg-slate-200 rounded-full overflow-hidden',
    bar: 'h-full bg-emerald-500 rounded-full transition-all duration-300 ease-out',
    percentage: 'text-right text-xs text-slate-400 dark:text-slate-600',
  },

  // Modal/Dialog
  modal: {
    overlay: 'fixed inset-0 bg-black/50 dark:bg-black/30 backdrop-blur-sm z-40',
    container: 'fixed inset-0 z-50 flex items-center justify-center p-4',
    content: 'bg-slate-900 dark:bg-white border border-slate-800 dark:border-slate-200 rounded-lg shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden',
    header: 'flex items-center justify-between p-6 border-b border-slate-800 dark:border-slate-200',
    body: 'p-6 overflow-y-auto',
    footer: 'flex items-center justify-end gap-3 p-6 border-t border-slate-800 dark:border-slate-200',
  },

  // Loading spinner
  spinner: 'inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 dark:border-slate-200 border-t-emerald-500',

  // Empty state
  emptyState: {
    container: 'text-center py-12',
    icon: 'inline-flex items-center justify-center w-16 h-16 rounded-full bg-slate-800 dark:bg-slate-100 mb-4',
    title: 'text-slate-400 dark:text-slate-600 text-lg',
    subtitle: 'text-slate-500 dark:text-slate-500 mt-2',
  },

  // Theme utility classes
  text: {
    primary: 'text-slate-100 dark:text-slate-900',
    secondary: 'text-slate-400 dark:text-slate-600',
    muted: 'text-slate-500 dark:text-slate-500',
    heading: 'text-slate-100 dark:text-slate-900',
  },

  surface: {
    base: 'bg-slate-900/50 dark:bg-white/90',
    elevated: 'bg-slate-900 dark:bg-white',
    card: 'bg-slate-800/50 dark:bg-slate-100',
    hover: 'hover:bg-slate-800/30 dark:hover:bg-slate-200/50',
  },

  border: {
    default: 'border-slate-800 dark:border-slate-200',
    muted: 'border-slate-700 dark:border-slate-300',
    hover: 'hover:border-slate-700 dark:hover:border-slate-400',
  },

  stat: {
    container: 'bg-slate-800/50 dark:bg-slate-100 rounded p-3',
    label: 'text-xs text-slate-400 dark:text-slate-600 mb-1',
    value: 'text-lg font-bold',
  },
} as const;

/**
 * Animation durations (in milliseconds)
 */
export const ANIMATION = {
  fast: 150,
  normal: 300,
  slow: 500,
} as const;

/**
 * Z-index layers
 */
export const Z_INDEX = {
  base: 0,
  dropdown: 10,
  sticky: 20,
  header: 30,
  overlay: 40,
  modal: 50,
  popover: 60,
  tooltip: 70,
} as const;

/**
 * Breakpoints for responsive design
 */
export const BREAKPOINTS = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
} as const;

/**
 * Common shadow styles
 */
export const SHADOWS = {
  sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
  default: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
  md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
  lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
  xl: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
  '2xl': '0 25px 50px -12px rgb(0 0 0 / 0.25)',
  inner: 'inset 0 2px 4px 0 rgb(0 0 0 / 0.05)',
  none: '0 0 #0000',
} as const;

/**
 * Utility function to combine className strings
 */
export const cx = (...classes: (string | boolean | undefined | null)[]) => {
  return classes.filter(Boolean).join(' ');
};
