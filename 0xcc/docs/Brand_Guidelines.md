# MechInterp Studio - Brand Guidelines

**Version:** 1.0
**Last Updated:** 2025-10-25
**Owner:** Product Design Team

---

## Table of Contents
1. [Brand Identity](#brand-identity)
2. [Logo Usage](#logo-usage)
3. [Color System](#color-system)
4. [Typography](#typography)
5. [Component Patterns](#component-patterns)
6. [Iconography](#iconography)
7. [Motion & Animation](#motion--animation)
8. [Voice & Tone](#voice--tone)
9. [Do's and Don'ts](#dos-and-donts)

---

## Brand Identity

### Overview
**MechInterp Studio** (miStudio) is a professional-grade tool for training and analyzing Sparse Autoencoders on edge AI hardware. Our brand reflects precision, technical excellence, and accessibility for ML researchers and engineers.

### Brand Attributes
- **Technical** - Built for AI researchers who demand precision
- **Modern** - Clean, contemporary interface design
- **Powerful** - Professional-grade feature set
- **Accessible** - Clear information hierarchy and intuitive workflows
- **Efficient** - Optimized for edge hardware constraints

### Brand Promise
*"Democratizing neural network interpretability with professional tools optimized for edge AI hardware."*

---

## Logo Usage

### Primary Logo
The primary logo consists of a **neural network pattern** with three connected nodes, representing the interpretability journey from model to features to insights.

**Files:**
- `frontend/src/assets/logo.svg` - 32x32 icon (primary)
- `frontend/src/assets/logo-wordmark.svg` - Horizontal wordmark
- `frontend/public/favicon.svg` - Favicon (simplified)

### Logo Anatomy
```
┌──────────────────────────┐
│  ◉━◉━◉  miStudio        │
│  └─ Neural nodes         │
│     └─ Emerald accent    │
└──────────────────────────┘
```

**Symbol Meaning:**
- **4 Nodes** = Input → Layer → Feature → Interpretation
- **Connection Lines** = Data flow and relationships
- **Emerald Accent** = Active discovery and insights
- **Dark Fill** = Technical foundation

### Clear Space
Maintain minimum clear space equal to the height of one node (4px) around all sides of the logo.

### Minimum Size
- **Digital:** 24px height minimum
- **Favicon:** 16px minimum (use simplified version)

### Logo Colors
- **Primary:** Emerald nodes (#10b981) on dark background (#0f1629)
- **Monochrome:** Slate-100 (#f1f5f9) on dark background
- **Inverted:** Dark nodes on light background (use sparingly)

### Incorrect Usage
❌ Do not rotate or skew the logo
❌ Do not change node colors arbitrarily
❌ Do not add effects (drop shadow, glow, gradient)
❌ Do not place on low-contrast backgrounds
❌ Do not separate nodes from wordmark in horizontal version

---

## Color System

### Philosophy
Our dark-first color system emphasizes **content over chrome** while maintaining excellent readability and visual hierarchy. The emerald accent provides energy and focus without overwhelming the technical data.

### Color Palette

#### Brand Colors (Primary)
```css
Emerald 500 (Primary)   #10b981  ███████  Main brand color, CTAs, active states
Emerald 600 (Hover)     #059669  ███████  Hover states, pressed buttons
Emerald 400 (Light)     #34d399  ███████  Highlights, secondary accents
Emerald 700 (Dark)      #047857  ███████  Deep accents, focus borders
```

#### Surface Colors (Backgrounds)
```css
Slate 950 (Base)        #0f1629  ███████  Main background (custom)
Slate 900 (Elevated)    #1e293b  ███████  Panel backgrounds, cards
Slate 800 (Card)        #334155  ███████  Nested cards, inputs
Slate 700 (Overlay)     #475569  ███████  Disabled states, overlays
```

#### Text Colors
```css
Slate 100 (Primary)     #f1f5f9  ███████  Headings, primary text
Slate 300 (Secondary)   #cbd5e1  ███████  Labels, secondary text
Slate 400 (Muted)       #94a3b8  ███████  Placeholder, helper text
Slate 500 (Disabled)    #64748b  ███████  Disabled text
```

#### Status Colors
```css
Success (Emerald)       #10b981  ███████  Completed, ready, positive
Warning (Amber)         #f59e0b  ███████  In-progress, caution
Error (Red)             #ef4444  ███████  Failed, critical, destructive
Info (Blue)             #3b82f6  ███████  Downloading, informational
Processing (Violet)     #8b5cf6  ███████  Active processing, computing
```

### Color Usage Guidelines

**Backgrounds:**
- Use `slate-950` as the base page background
- Use `slate-900` for elevated panels and cards
- Use `slate-800` for input fields and nested components
- Apply transparency (`/50`, `/30`) for subtle overlays

**Text:**
- Use `slate-100` for primary headings and important text
- Use `slate-300` for form labels and secondary content
- Use `slate-400` for placeholder text and subtle information
- Never use pure white (`#ffffff`) - maintain dark theme consistency

**Accents:**
- Use emerald colors for **interactive elements only** (buttons, links, focus states)
- Use status colors **semantically** (don't use red for non-error states)
- Maintain 4.5:1 contrast ratio minimum for WCAG AA compliance

**Borders:**
- Default borders: `slate-800`
- Hover borders: `slate-700`
- Focus borders: `emerald-500` with 2px ring

### Accessibility
All color combinations meet WCAG 2.1 AA standards (4.5:1 contrast) for normal text and AAA standards (7:1) for large text.

**Contrast Ratios:**
- `slate-100` on `slate-950`: **14.2:1** ✅ AAA
- `slate-300` on `slate-900`: **7.8:1** ✅ AAA
- `emerald-400` on `slate-950`: **6.1:1** ✅ AA+
- `slate-400` on `slate-800`: **4.7:1** ✅ AA

---

## Typography

### Font Stack
```css
/* Headings & Body */
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
             'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
             sans-serif;

/* Code & Technical Values */
font-family: 'SF Mono', Monaco, 'Cascadia Code', Consolas, 'Courier New', monospace;
```

**Rationale:** System fonts provide excellent performance, native appearance, and consistent rendering across platforms.

### Type Scale
| Size | Token | rem | px | Usage |
|------|-------|-----|----|----|
| xs | text-xs | 0.75rem | 12px | Badges, labels, timestamps |
| sm | text-sm | 0.875rem | 14px | Secondary text, captions |
| base | text-base | 1rem | 16px | Body text, paragraphs |
| lg | text-lg | 1.125rem | 18px | Section headers |
| xl | text-xl | 1.25rem | 20px | Panel titles |
| 2xl | text-2xl | 1.5rem | 24px | Page headers |
| 3xl | text-3xl | 1.875rem | 30px | Hero headings |

### Font Weights
| Weight | Token | Value | Usage |
|--------|-------|-------|-------|
| Normal | font-normal | 400 | Body text, paragraphs |
| Medium | font-medium | 500 | Labels, navigation |
| Semibold | font-semibold | 600 | Headings, emphasis |
| Bold | font-bold | 700 | Strong emphasis (rare) |

### Typographic Hierarchy

**Page Structure:**
```tsx
// Page Header (h1)
<h1 className="text-2xl font-semibold text-slate-100 mb-2">
  Page Title
</h1>
<p className="text-slate-400">Descriptive subtitle</p>

// Section Header (h2)
<h2 className="text-lg font-semibold text-slate-100 mb-4">
  Section Title
</h2>

// Subsection (h3)
<h3 className="text-base font-medium text-slate-300 mb-2">
  Subsection
</h3>

// Body Text
<p className="text-sm text-slate-300">
  Regular paragraph text
</p>

// Helper Text
<p className="text-xs text-slate-400">
  Helper or caption text
</p>
```

### Code & Technical Text
Use monospace font for:
- Model IDs and identifiers
- File paths
- Numeric values in data tables
- Configuration snippets
- Command-line examples

```tsx
<code className="font-mono text-sm text-emerald-400">
  gpt2-medium
</code>
```

---

## Component Patterns

### Design Tokens
All components should use design tokens from `frontend/src/config/brand.ts`:

```typescript
import { COMPONENTS, COLORS, TYPOGRAPHY } from '@/config/brand';

// Use predefined component styles
<button className={COMPONENTS.button.primary}>
  Primary Action
</button>
```

### Cards
Cards are the primary container element for grouped content.

**Base Card:**
```tsx
<div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
  Card content
</div>
```

**Elevated Card:**
```tsx
<div className="bg-slate-900 border border-slate-700 rounded-lg shadow-lg p-6">
  Elevated content
</div>
```

**Interactive Card:**
```tsx
<div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6
                hover:border-slate-700 transition-colors cursor-pointer">
  Clickable content
</div>
```

### Buttons

**Primary (High Emphasis):**
```tsx
<button className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700
                   text-white font-medium rounded transition-colors">
  Primary Action
</button>
```

**Secondary (Medium Emphasis):**
```tsx
<button className="px-4 py-2 bg-slate-800 hover:bg-slate-700
                   text-slate-300 font-medium rounded transition-colors">
  Secondary Action
</button>
```

**Ghost (Low Emphasis):**
```tsx
<button className="px-4 py-2 hover:bg-slate-800 text-slate-400
                   hover:text-slate-300 rounded transition-colors">
  Tertiary Action
</button>
```

**Danger (Destructive):**
```tsx
<button className="px-4 py-2 bg-red-600 hover:bg-red-700
                   text-white font-medium rounded transition-colors">
  Delete
</button>
```

### Form Inputs

**Text Input:**
```tsx
<input className="w-full px-4 py-2 bg-slate-800 border border-slate-700
                  rounded text-slate-100 placeholder-slate-500
                  focus:outline-none focus:ring-2 focus:ring-emerald-500
                  focus:border-transparent" />
```

**Textarea:**
```tsx
<textarea className="w-full px-4 py-2 bg-slate-800 border border-slate-700
                     rounded text-slate-100 placeholder-slate-500
                     focus:outline-none focus:ring-2 focus:ring-emerald-500
                     focus:border-transparent resize-none" />
```

**Select/Dropdown:**
```tsx
<select className="w-full px-4 py-2 bg-slate-800 border border-slate-700
                   rounded text-slate-100 focus:outline-none focus:ring-2
                   focus:ring-emerald-500 focus:border-transparent">
  <option>Option 1</option>
</select>
```

### Status Badges

```tsx
// Success
<span className="inline-flex items-center px-3 py-1 rounded-full text-xs
                 font-medium border bg-emerald-500/20 text-emerald-400
                 border-emerald-500/30">
  Ready
</span>

// Warning
<span className="inline-flex items-center px-3 py-1 rounded-full text-xs
                 font-medium border bg-yellow-500/20 text-yellow-400
                 border-yellow-500/30">
  Processing
</span>

// Error
<span className="inline-flex items-center px-3 py-1 rounded-full text-xs
                 font-medium border bg-red-500/20 text-red-400
                 border-red-500/30">
  Failed
</span>
```

### Progress Indicators

**Progress Bar:**
```tsx
<div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
  <div className="h-full bg-emerald-500 rounded-full transition-all
                  duration-300 ease-out"
       style={{ width: `${progress}%` }} />
</div>
```

**Loading Spinner:**
```tsx
<div className="inline-block animate-spin rounded-full h-8 w-8 border-4
                border-slate-700 border-t-emerald-500" />
```

### Modals/Dialogs

```tsx
// Overlay
<div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40" />

// Modal Container
<div className="fixed inset-0 z-50 flex items-center justify-center p-4">
  <div className="bg-slate-900 border border-slate-800 rounded-lg shadow-2xl
                  max-w-2xl w-full max-h-[90vh] overflow-hidden">
    {/* Header */}
    <div className="flex items-center justify-between p-6 border-b border-slate-800">
      <h2 className="text-xl font-semibold text-slate-100">Modal Title</h2>
      <button className="p-1 hover:bg-slate-800 rounded">×</button>
    </div>

    {/* Body */}
    <div className="p-6 overflow-y-auto">
      Content
    </div>

    {/* Footer */}
    <div className="flex items-center justify-end gap-3 p-6 border-t border-slate-800">
      <button>Cancel</button>
      <button>Confirm</button>
    </div>
  </div>
</div>
```

### Empty States

```tsx
<div className="text-center py-12">
  <div className="inline-flex items-center justify-center w-16 h-16
                  rounded-full bg-slate-800 mb-4">
    <Icon className="h-8 w-8 text-emerald-500/50" />
  </div>
  <p className="text-slate-400 text-lg">No items yet</p>
  <p className="text-slate-500 mt-2">Get started by adding your first item</p>
</div>
```

---

## Iconography

### Icon Library
We use **Lucide React** for all icons - a clean, consistent, MIT-licensed icon set.

**Installation:**
```bash
npm install lucide-react
```

**Usage:**
```tsx
import { Download, Trash2, Settings } from 'lucide-react';

<Download className="h-5 w-5 text-slate-400" />
```

### Icon Sizing
| Size |ClassName | Pixels | Usage |
|------|-----------|--------|-------|
| Small | h-4 w-4 | 16px | Inline text icons |
| Default | h-5 w-5 | 20px | Button icons, list items |
| Medium | h-6 w-6 | 24px | Feature icons, headers |
| Large | h-8 w-8 | 32px | Empty states, hero sections |

### Icon Colors
- **Default:** `text-slate-400` (muted)
- **Hover:** `text-slate-300` (lighter)
- **Active/Primary:** `text-emerald-500` (brand)
- **Success:** `text-emerald-400`
- **Warning:** `text-yellow-400`
- **Error:** `text-red-400`

### Icon Guidelines
✅ Use outline style icons (not filled) for consistency
✅ Pair icons with text labels for clarity
✅ Maintain 16px minimum touch target for interactive icons
✅ Use semantic icons (trash for delete, download for download)

❌ Don't mix icon styles (outline + filled)
❌ Don't use icons alone without labels for complex actions
❌ Don't resize icons non-proportionally

---

## Motion & Animation

### Animation Principles
- **Purposeful** - Animations should provide feedback or guide attention
- **Subtle** - Avoid distracting users from their work
- **Fast** - Keep durations under 300ms for UI interactions
- **Consistent** - Use standard easing functions

### Duration Standards
```typescript
// From brand.ts
ANIMATION = {
  fast: 150,      // Hover states, simple transitions
  normal: 300,    // Modal open/close, slide transitions
  slow: 500,      // Complex animations, page transitions
}
```

### Common Transitions

**Hover States:**
```css
transition-colors  /* 150ms default */
```

**Modals/Overlays:**
```css
transition-opacity duration-300
```

**Slide Animations:**
```css
transition-transform duration-300 ease-out
```

### Animation Classes
```tsx
// Fade in
<div className="animate-fade-in">

// Spin (loading)
<div className="animate-spin">

// Slide up
<div className="animate-slide-up">
```

### Reduced Motion
Always respect user's motion preferences:

```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## Voice & Tone

### Brand Voice
**Professional yet approachable** - We speak to ML researchers and engineers who value precision and clarity.

**Characteristics:**
- **Clear** - Use simple, direct language
- **Technical** - Don't dumb down ML concepts
- **Helpful** - Guide users, don't gatekeep
- **Confident** - Assert capabilities without arrogance

### Writing Guidelines

**✅ Do:**
- Use active voice ("Train a model" not "A model can be trained")
- Be specific with technical terms ("Sparse Autoencoder" not "AI model")
- Provide context in error messages ("Failed to load dataset: Connection timeout" not "Error 500")
- Use sentence case for UI labels ("Download from HuggingFace" not "Download From HuggingFace")

**❌ Don't:**
- Use jargon without explanation
- Anthropomorphize the system ("I think..." - the system doesn't think)
- Be cutesy or overly casual
- Use ALL CAPS for emphasis (use bold or color)

### Microcopy Examples

**Empty States:**
```
✅ "No datasets yet. Download a dataset from HuggingFace to get started."
❌ "Oops! Nothing here! 😢"
```

**Error Messages:**
```
✅ "Failed to download model: Network connection lost. Please check your connection and try again."
❌ "Something went wrong!"
```

**Button Labels:**
```
✅ "Start Training" / "Download Dataset" / "Save Configuration"
❌ "Go" / "Submit" / "OK"
```

**Success Messages:**
```
✅ "Training completed successfully. View results →"
❌ "Yay! All done!"
```

---

## Do's and Don'ts

### Logo
✅ Use the provided SVG files
✅ Maintain minimum clear space
✅ Use on dark backgrounds primarily
✅ Keep nodes and wordmark together

❌ Don't rotate or distort
❌ Don't change colors arbitrarily
❌ Don't add effects or outlines
❌ Don't use on low-contrast backgrounds

### Color
✅ Use emerald for interactive elements
✅ Maintain color hierarchy (primary/secondary/muted)
✅ Use status colors semantically
✅ Test contrast ratios

❌ Don't use pure white text
❌ Don't use emerald for non-interactive elements
❌ Don't use red/green for non-status meanings
❌ Don't ignore accessibility guidelines

### Typography
✅ Use system font stack
✅ Maintain type hierarchy
✅ Use monospace for technical values
✅ Keep line length readable (60-80 chars)

❌ Don't use more than 3 font sizes on a page
❌ Don't use custom fonts (performance impact)
❌ Don't center-align large blocks of text
❌ Don't use font size alone for hierarchy

### Components
✅ Use design tokens from `brand.ts`
✅ Maintain consistent spacing
✅ Follow established patterns
✅ Test with real content

❌ Don't create one-off styles
❌ Don't use inline styles
❌ Don't ignore disabled states
❌ Don't skip focus indicators

---

## Implementation Checklist

When implementing a new feature or component:

- [ ] Use colors from `brand.ts` COLORS
- [ ] Use component presets from `brand.ts` COMPONENTS
- [ ] Follow typography hierarchy
- [ ] Include hover and focus states
- [ ] Add loading and empty states
- [ ] Test keyboard navigation
- [ ] Verify color contrast (4.5:1 minimum)
- [ ] Test with reduced motion preferences
- [ ] Use semantic HTML elements
- [ ] Add ARIA labels where needed

---

## Resources

### Design Files
- Logo SVG: `frontend/src/assets/logo.svg`
- Wordmark: `frontend/src/assets/logo-wordmark.svg`
- Favicon: `frontend/public/favicon.svg`

### Code
- Design Tokens: `frontend/src/config/brand.ts`
- Tailwind Config: `frontend/tailwind.config.js`

### External References
- Tailwind CSS Docs: https://tailwindcss.com/docs
- Lucide Icons: https://lucide.dev/icons
- WCAG Guidelines: https://www.w3.org/WAI/WCAG21/quickref/

---

## Changelog

### Version 1.0 (2025-10-25)
- Initial brand guidelines creation
- Logo design and implementation
- Color system formalization
- Component pattern documentation
- Typography standards
- Accessibility guidelines

---

**Questions or feedback?** Open an issue in the repository or contact the design team.

**License:** This brand guide is part of the MechInterp Studio project and follows the same license.
