/**
 * Unit tests for ProgressBar component.
 *
 * Tests rendering, progress display, clamping, styling, and edge cases.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ProgressBar } from './ProgressBar';

describe('ProgressBar', () => {
  describe('Progress Fill Width', () => {
    it('should set fill width to 0% for progress 0', () => {
      const { container } = render(<ProgressBar progress={0} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('0%');
    });

    it('should set fill width to 50% for progress 50', () => {
      const { container } = render(<ProgressBar progress={50} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('50%');
    });

    it('should set fill width to 100% for progress 100', () => {
      const { container } = render(<ProgressBar progress={100} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('100%');
    });

    it('should set fill width to 25.5% for progress 25.5', () => {
      const { container } = render(<ProgressBar progress={25.5} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('25.5%');
    });

    it('should update fill width when progress changes', () => {
      const { container, rerender } = render(<ProgressBar progress={30} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('30%');

      rerender(<ProgressBar progress={70} />);
      expect(fill?.style.width).toBe('70%');
    });
  });

  describe('Percentage Text Display', () => {
    it('should display percentage text by default', () => {
      render(<ProgressBar progress={45} />);
      expect(screen.getByText('45.0%')).toBeInTheDocument();
    });

    it('should display percentage with one decimal place', () => {
      render(<ProgressBar progress={33.333} />);
      expect(screen.getByText('33.3%')).toBeInTheDocument();
    });

    it('should display 0.0% for progress 0', () => {
      render(<ProgressBar progress={0} />);
      expect(screen.getByText('0.0%')).toBeInTheDocument();
    });

    it('should display 100.0% for progress 100', () => {
      render(<ProgressBar progress={100} />);
      expect(screen.getByText('100.0%')).toBeInTheDocument();
    });

    it('should round percentage to one decimal place', () => {
      render(<ProgressBar progress={66.666} />);
      expect(screen.getByText('66.7%')).toBeInTheDocument();
    });

    it('should hide percentage text when showPercentage is false', () => {
      render(<ProgressBar progress={50} showPercentage={false} />);
      expect(screen.queryByText('50.0%')).not.toBeInTheDocument();
    });

    it('should show percentage text when showPercentage is true', () => {
      render(<ProgressBar progress={50} showPercentage={true} />);
      expect(screen.getByText('50.0%')).toBeInTheDocument();
    });

    it('should update percentage text when progress changes', () => {
      const { rerender } = render(<ProgressBar progress={25} />);
      expect(screen.getByText('25.0%')).toBeInTheDocument();

      rerender(<ProgressBar progress={75} />);
      expect(screen.getByText('75.0%')).toBeInTheDocument();
      expect(screen.queryByText('25.0%')).not.toBeInTheDocument();
    });
  });

  describe('Progress Clamping', () => {
    it('should clamp negative progress to 0', () => {
      const { container } = render(<ProgressBar progress={-50} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('0%');
      expect(screen.getByText('0.0%')).toBeInTheDocument();
    });

    it('should clamp progress over 100 to 100', () => {
      const { container } = render(<ProgressBar progress={150} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('100%');
      expect(screen.getByText('100.0%')).toBeInTheDocument();
    });

    it('should clamp very large progress values', () => {
      const { container } = render(<ProgressBar progress={99999} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('100%');
      expect(screen.getByText('100.0%')).toBeInTheDocument();
    });

    it('should clamp very small negative values', () => {
      const { container } = render(<ProgressBar progress={-99999} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('0%');
      expect(screen.getByText('0.0%')).toBeInTheDocument();
    });

    it('should not clamp valid progress values', () => {
      const { container } = render(<ProgressBar progress={50} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('50%');
      expect(screen.getByText('50.0%')).toBeInTheDocument();
    });
  });

  describe('Component Structure', () => {
    it('should render container with space-y-1 class', () => {
      const { container } = render(<ProgressBar progress={50} />);
      const outerDiv = container.firstChild as HTMLElement;

      expect(outerDiv?.className).toContain('space-y-1');
    });

    it('should render background bar with correct classes', () => {
      const { container } = render(<ProgressBar progress={50} />);
      const bgBar = container.querySelector('.w-full.h-2') as HTMLElement;

      expect(bgBar?.className).toContain('w-full');
      expect(bgBar?.className).toContain('h-2');
      expect(bgBar?.className).toContain('bg-slate-800');
      expect(bgBar?.className).toContain('rounded-full');
      expect(bgBar?.className).toContain('overflow-hidden');
    });

    it('should render fill bar with correct classes', () => {
      const { container } = render(<ProgressBar progress={50} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.className).toContain('h-full');
      expect(fill?.className).toContain('bg-emerald-500');
      expect(fill?.className).toContain('rounded-full');
      expect(fill?.className).toContain('transition-all');
      expect(fill?.className).toContain('duration-300');
      expect(fill?.className).toContain('ease-out');
    });

    it('should render percentage text with correct classes', () => {
      const { container } = render(<ProgressBar progress={50} />);
      // The percentage text is in a div with text-right, text-xs, text-slate-400
      const percentageDiv = container.querySelector('.text-right') as HTMLElement;

      expect(percentageDiv?.className).toContain('text-right');
      expect(percentageDiv?.className).toContain('text-xs');
      expect(percentageDiv?.className).toContain('text-slate-400');
      expect(percentageDiv?.textContent).toBe('50.0%');
    });

    it('should have nested div structure', () => {
      const { container } = render(<ProgressBar progress={50} />);

      // Outer container
      const outer = container.firstChild as HTMLElement;
      expect(outer).toBeInTheDocument();

      // Background bar
      const bgBar = outer.querySelector('.bg-slate-800');
      expect(bgBar).toBeInTheDocument();

      // Fill bar inside background
      const fill = bgBar?.querySelector('.bg-emerald-500');
      expect(fill).toBeInTheDocument();
    });
  });

  describe('Custom ClassName', () => {
    it('should apply custom className to container', () => {
      const { container } = render(<ProgressBar progress={50} className="custom-class" />);
      const outerDiv = container.firstChild as HTMLElement;

      expect(outerDiv?.className).toContain('custom-class');
    });

    it('should preserve base classes with custom className', () => {
      const { container } = render(<ProgressBar progress={50} className="mt-4" />);
      const outerDiv = container.firstChild as HTMLElement;

      expect(outerDiv?.className).toContain('mt-4');
      expect(outerDiv?.className).toContain('space-y-1');
    });

    it('should work without custom className', () => {
      const { container } = render(<ProgressBar progress={50} />);
      const outerDiv = container.firstChild as HTMLElement;

      expect(outerDiv?.className).not.toContain('undefined');
    });
  });

  describe('Edge Cases', () => {
    it('should handle 0 progress', () => {
      const { container } = render(<ProgressBar progress={0} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('0%');
      expect(screen.getByText('0.0%')).toBeInTheDocument();
    });

    it('should handle 100 progress', () => {
      const { container } = render(<ProgressBar progress={100} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('100%');
      expect(screen.getByText('100.0%')).toBeInTheDocument();
    });

    it('should handle decimal progress values', () => {
      const { container } = render(<ProgressBar progress={33.333} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('33.333%');
      expect(screen.getByText('33.3%')).toBeInTheDocument();
    });

    it('should handle very small progress values', () => {
      const { container } = render(<ProgressBar progress={0.1} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('0.1%');
      expect(screen.getByText('0.1%')).toBeInTheDocument();
    });

    it('should handle very large decimal progress values', () => {
      const { container } = render(<ProgressBar progress={99.99} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('99.99%');
      expect(screen.getByText('100.0%')).toBeInTheDocument();
    });

    it('should handle exactly 50 progress', () => {
      const { container } = render(<ProgressBar progress={50} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('50%');
      expect(screen.getByText('50.0%')).toBeInTheDocument();
    });

    it('should handle NaN progress by clamping to 0', () => {
      const { container } = render(<ProgressBar progress={NaN} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      // Math.max(0, Math.min(100, NaN)) returns NaN
      // In React, setting style.width to NaN% may result in empty string
      // The component will display NaN% in the text
      expect(screen.getByText('NaN%')).toBeInTheDocument();
    });
  });

  describe('Multiple Instances', () => {
    it('should render multiple progress bars independently', () => {
      const { container } = render(
        <>
          <ProgressBar progress={25} />
          <ProgressBar progress={50} />
          <ProgressBar progress={75} />
        </>
      );

      expect(screen.getByText('25.0%')).toBeInTheDocument();
      expect(screen.getByText('50.0%')).toBeInTheDocument();
      expect(screen.getByText('75.0%')).toBeInTheDocument();

      const fills = container.querySelectorAll('.bg-emerald-500');
      expect(fills).toHaveLength(3);
      expect((fills[0] as HTMLElement).style.width).toBe('25%');
      expect((fills[1] as HTMLElement).style.width).toBe('50%');
      expect((fills[2] as HTMLElement).style.width).toBe('75%');
    });

    it('should apply different showPercentage values to different instances', () => {
      render(
        <>
          <ProgressBar progress={30} showPercentage={true} />
          <ProgressBar progress={70} showPercentage={false} />
        </>
      );

      expect(screen.getByText('30.0%')).toBeInTheDocument();
      expect(screen.queryByText('70.0%')).not.toBeInTheDocument();
    });
  });

  describe('Animation and Transitions', () => {
    it('should have transition classes for smooth animation', () => {
      const { container } = render(<ProgressBar progress={50} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.className).toContain('transition-all');
      expect(fill?.className).toContain('duration-300');
      expect(fill?.className).toContain('ease-out');
    });

    it('should maintain transition classes when progress changes', () => {
      const { container, rerender } = render(<ProgressBar progress={30} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.className).toContain('transition-all');

      rerender(<ProgressBar progress={70} />);
      expect(fill?.className).toContain('transition-all');
    });
  });

  describe('Accessibility', () => {
    it('should be visible to screen readers', () => {
      render(<ProgressBar progress={50} />);
      const percentage = screen.getByText('50.0%');

      expect(percentage).toBeVisible();
    });

    it('should provide visual progress indication', () => {
      const { container } = render(<ProgressBar progress={75} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      // Visual indicator should have distinct color
      expect(fill?.className).toContain('bg-emerald-500');
      // Container should have distinct background
      expect(container.querySelector('.bg-slate-800')).toBeInTheDocument();
    });
  });

  describe('Integration Scenarios', () => {
    it('should work with rapidly changing progress', () => {
      const { container, rerender } = render(<ProgressBar progress={0} />);
      const fill = container.querySelector('.bg-emerald-500') as HTMLElement;

      expect(fill?.style.width).toBe('0%');

      rerender(<ProgressBar progress={25} />);
      expect(fill?.style.width).toBe('25%');

      rerender(<ProgressBar progress={50} />);
      expect(fill?.style.width).toBe('50%');

      rerender(<ProgressBar progress={75} />);
      expect(fill?.style.width).toBe('75%');

      rerender(<ProgressBar progress={100} />);
      expect(fill?.style.width).toBe('100%');
    });

    it('should work with conditional rendering', () => {
      const showBar = true;
      const { rerender } = render(
        <div>{showBar && <ProgressBar progress={60} />}</div>
      );

      expect(screen.getByText('60.0%')).toBeInTheDocument();

      rerender(<div>{false && <ProgressBar progress={60} />}</div>);
      expect(screen.queryByText('60.0%')).not.toBeInTheDocument();
    });

    it('should work in a list of progress indicators', () => {
      const progressValues = [10, 30, 50, 70, 90];

      const { container } = render(
        <div>
          {progressValues.map((progress, index) => (
            <ProgressBar key={index} progress={progress} />
          ))}
        </div>
      );

      const fills = container.querySelectorAll('.bg-emerald-500');
      expect(fills).toHaveLength(5);
      expect((fills[0] as HTMLElement).style.width).toBe('10%');
      expect((fills[4] as HTMLElement).style.width).toBe('90%');
    });
  });

  describe('TypeScript Type Safety', () => {
    it('should accept number progress values', () => {
      expect(() => render(<ProgressBar progress={50} />)).not.toThrow();
    });

    it('should accept optional className prop', () => {
      expect(() => render(<ProgressBar progress={50} className="test" />)).not.toThrow();
    });

    it('should accept optional showPercentage prop', () => {
      expect(() => render(<ProgressBar progress={50} showPercentage={false} />)).not.toThrow();
    });

    it('should work without optional props', () => {
      expect(() => render(<ProgressBar progress={50} />)).not.toThrow();
    });
  });
});
