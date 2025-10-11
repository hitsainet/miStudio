/**
 * Unit tests for StatusBadge component.
 *
 * Tests rendering, status color mapping, text formatting, and edge cases.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusBadge } from './StatusBadge';
import { DatasetStatus } from '../../types/dataset';

describe('StatusBadge', () => {
  describe('Status Text Display', () => {
    it('should render "Downloading" for downloading status', () => {
      render(<StatusBadge status={DatasetStatus.DOWNLOADING} />);
      expect(screen.getByText('Downloading')).toBeInTheDocument();
    });

    it('should render "Processing" for processing status', () => {
      render(<StatusBadge status={DatasetStatus.PROCESSING} />);
      expect(screen.getByText('Processing')).toBeInTheDocument();
    });

    it('should render "Ready" for ready status', () => {
      render(<StatusBadge status={DatasetStatus.READY} />);
      expect(screen.getByText('Ready')).toBeInTheDocument();
    });

    it('should render "Error" for error status', () => {
      render(<StatusBadge status={DatasetStatus.ERROR} />);
      expect(screen.getByText('Error')).toBeInTheDocument();
    });

    it('should capitalize first letter of status text', () => {
      render(<StatusBadge status="downloading" />);
      const badge = screen.getByText('Downloading');
      expect(badge).toBeInTheDocument();
      expect(badge.textContent?.charAt(0)).toBe('D');
    });
  });

  describe('Color Mapping', () => {
    it('should apply blue color classes for downloading status', () => {
      const { container } = render(<StatusBadge status="downloading" />);
      const badge = container.querySelector('span');

      expect(badge?.className).toContain('bg-blue-500/20');
      expect(badge?.className).toContain('text-blue-400');
      expect(badge?.className).toContain('border-blue-500/30');
    });

    it('should apply yellow color classes for processing status', () => {
      const { container } = render(<StatusBadge status="processing" />);
      const badge = container.querySelector('span');

      expect(badge?.className).toContain('bg-yellow-500/20');
      expect(badge?.className).toContain('text-yellow-400');
      expect(badge?.className).toContain('border-yellow-500/30');
    });

    it('should apply emerald color classes for ready status', () => {
      const { container } = render(<StatusBadge status="ready" />);
      const badge = container.querySelector('span');

      expect(badge?.className).toContain('bg-emerald-500/20');
      expect(badge?.className).toContain('text-emerald-400');
      expect(badge?.className).toContain('border-emerald-500/30');
    });

    it('should apply red color classes for error status', () => {
      const { container } = render(<StatusBadge status="error" />);
      const badge = container.querySelector('span');

      expect(badge?.className).toContain('bg-red-500/20');
      expect(badge?.className).toContain('text-red-400');
      expect(badge?.className).toContain('border-red-500/30');
    });

    it('should use ready color as default for unknown status', () => {
      const { container } = render(<StatusBadge status="unknown" />);
      const badge = container.querySelector('span');

      expect(badge?.className).toContain('bg-emerald-500/20');
      expect(badge?.className).toContain('text-emerald-400');
      expect(badge?.className).toContain('border-emerald-500/30');
    });
  });

  describe('Base Styling', () => {
    it('should have base badge classes', () => {
      const { container } = render(<StatusBadge status="ready" />);
      const badge = container.querySelector('span');

      expect(badge?.className).toContain('inline-flex');
      expect(badge?.className).toContain('items-center');
      expect(badge?.className).toContain('px-3');
      expect(badge?.className).toContain('py-1');
      expect(badge?.className).toContain('rounded-full');
      expect(badge?.className).toContain('text-xs');
      expect(badge?.className).toContain('font-medium');
      expect(badge?.className).toContain('border');
    });

    it('should render as span element', () => {
      const { container } = render(<StatusBadge status="ready" />);
      const badge = container.querySelector('span');

      expect(badge?.tagName).toBe('SPAN');
    });
  });

  describe('Custom ClassName', () => {
    it('should apply custom className when provided', () => {
      const { container } = render(<StatusBadge status="ready" className="custom-class" />);
      const badge = container.querySelector('span');

      expect(badge?.className).toContain('custom-class');
    });

    it('should preserve base classes with custom className', () => {
      const { container } = render(<StatusBadge status="ready" className="ml-2" />);
      const badge = container.querySelector('span');

      expect(badge?.className).toContain('ml-2');
      expect(badge?.className).toContain('inline-flex');
      expect(badge?.className).toContain('px-3');
    });

    it('should work without custom className', () => {
      const { container } = render(<StatusBadge status="ready" />);
      const badge = container.querySelector('span');

      expect(badge?.className).not.toContain('undefined');
    });
  });

  describe('Status Normalization', () => {
    it('should handle uppercase status strings', () => {
      render(<StatusBadge status="DOWNLOADING" />);
      expect(screen.getByText('Downloading')).toBeInTheDocument();
    });

    it('should handle mixed case status strings', () => {
      render(<StatusBadge status="DoWnLoAdInG" />);
      expect(screen.getByText('Downloading')).toBeInTheDocument();
    });

    it('should apply correct colors after normalizing case', () => {
      const { container } = render(<StatusBadge status="READY" />);
      const badge = container.querySelector('span');

      expect(badge?.className).toContain('bg-emerald-500/20');
    });

    it('should handle DatasetStatus enum values', () => {
      render(<StatusBadge status={DatasetStatus.DOWNLOADING} />);
      expect(screen.getByText('Downloading')).toBeInTheDocument();
    });

    it('should handle string literal status values', () => {
      render(<StatusBadge status="downloading" />);
      expect(screen.getByText('Downloading')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty string status', () => {
      const { container } = render(<StatusBadge status="" />);
      const badge = container.querySelector('span');

      // Empty string capitalizes to empty string
      expect(badge?.textContent).toBe('');
      // Should use default (ready) color
      expect(badge?.className).toContain('bg-emerald-500/20');
    });

    it('should handle single character status', () => {
      render(<StatusBadge status="r" />);
      expect(screen.getByText('R')).toBeInTheDocument();
    });

    it('should handle status with spaces', () => {
      render(<StatusBadge status="in progress" />);
      expect(screen.getByText('In progress')).toBeInTheDocument();
    });

    it('should handle status with hyphens', () => {
      render(<StatusBadge status="not-found" />);
      expect(screen.getByText('Not-found')).toBeInTheDocument();
    });

    it('should handle status with underscores', () => {
      render(<StatusBadge status="in_progress" />);
      expect(screen.getByText('In_progress')).toBeInTheDocument();
    });

    it('should handle very long status strings', () => {
      const longStatus = 'this-is-a-very-long-status-string-that-should-still-work';
      render(<StatusBadge status={longStatus} />);
      expect(screen.getByText(/This-is-a-very-long/)).toBeInTheDocument();
    });

    it('should handle numeric-like status strings', () => {
      render(<StatusBadge status="404" />);
      expect(screen.getByText('404')).toBeInTheDocument();
    });

    it('should handle special characters in status', () => {
      render(<StatusBadge status="error!" />);
      expect(screen.getByText('Error!')).toBeInTheDocument();
    });
  });

  describe('Multiple Instances', () => {
    it('should render multiple badges independently', () => {
      const { container } = render(
        <>
          <StatusBadge status="downloading" />
          <StatusBadge status="ready" />
          <StatusBadge status="error" />
        </>
      );

      expect(screen.getByText('Downloading')).toBeInTheDocument();
      expect(screen.getByText('Ready')).toBeInTheDocument();
      expect(screen.getByText('Error')).toBeInTheDocument();

      const badges = container.querySelectorAll('span');
      expect(badges).toHaveLength(3);
    });

    it('should apply different colors to different statuses', () => {
      const { container } = render(
        <>
          <StatusBadge status="downloading" />
          <StatusBadge status="ready" />
        </>
      );

      const badges = container.querySelectorAll('span');
      expect(badges[0].className).toContain('bg-blue-500/20');
      expect(badges[1].className).toContain('bg-emerald-500/20');
    });
  });

  describe('Accessibility', () => {
    it('should be readable by screen readers', () => {
      render(<StatusBadge status="downloading" />);
      const badge = screen.getByText('Downloading');

      expect(badge).toBeVisible();
    });

    it('should have appropriate contrast with colored backgrounds', () => {
      // This test verifies classes are applied; actual contrast testing
      // would require visual regression testing
      const { container } = render(<StatusBadge status="downloading" />);
      const badge = container.querySelector('span');

      // Verify text color is set (important for contrast)
      expect(badge?.className).toContain('text-blue-400');
    });
  });

  describe('Integration Scenarios', () => {
    it('should work within a list of statuses', () => {
      const statuses: DatasetStatus[] = [
        DatasetStatus.DOWNLOADING,
        DatasetStatus.PROCESSING,
        DatasetStatus.READY,
        DatasetStatus.ERROR,
      ];

      const { container } = render(
        <div>
          {statuses.map((status, index) => (
            <StatusBadge key={index} status={status} />
          ))}
        </div>
      );

      const badges = container.querySelectorAll('span');
      expect(badges).toHaveLength(4);
    });

    it('should work with conditional rendering', () => {
      const showBadge = true;
      const { container, rerender } = render(
        <div>{showBadge && <StatusBadge status="ready" />}</div>
      );

      expect(screen.getByText('Ready')).toBeInTheDocument();

      rerender(<div>{false && <StatusBadge status="ready" />}</div>);
      expect(screen.queryByText('Ready')).not.toBeInTheDocument();
    });

    it('should update when status prop changes', () => {
      const { rerender } = render(<StatusBadge status="downloading" />);

      expect(screen.getByText('Downloading')).toBeInTheDocument();

      rerender(<StatusBadge status="ready" />);

      expect(screen.queryByText('Downloading')).not.toBeInTheDocument();
      expect(screen.getByText('Ready')).toBeInTheDocument();
    });
  });

  describe('TypeScript Type Safety', () => {
    it('should accept DatasetStatus enum values', () => {
      expect(() => render(<StatusBadge status={DatasetStatus.READY} />)).not.toThrow();
    });

    it('should accept string values', () => {
      expect(() => render(<StatusBadge status="custom-status" />)).not.toThrow();
    });

    it('should accept optional className prop', () => {
      expect(() => render(<StatusBadge status="ready" className="test" />)).not.toThrow();
    });

    it('should work without className prop', () => {
      expect(() => render(<StatusBadge status="ready" />)).not.toThrow();
    });
  });
});
