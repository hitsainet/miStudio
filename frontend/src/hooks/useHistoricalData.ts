/**
 * useHistoricalData Hook
 *
 * Manages time-series data storage for GPU and system metrics.
 * Stores data points with timestamps and provides aggregation for different time ranges.
 */

import { useState, useCallback, useRef } from 'react';

export type TimeRange = '1h' | '6h' | '24h';

export interface DataPoint {
  timestamp: number;
  value: number;
}

export interface MultiSeriesDataPoint {
  timestamp: number;
  [key: string]: number;
}

interface HistoricalDataOptions {
  maxDataPoints?: number;
  aggregationInterval?: number; // in seconds
}

export function useHistoricalData(_options: HistoricalDataOptions = {}) {
  const [data, setData] = useState<MultiSeriesDataPoint[]>([]);
  const timeRange: TimeRange = '1h'; // Fixed to 1 hour
  const lastPruneTime = useRef<number>(Date.now());

  // Time range configurations (in milliseconds)
  const timeRangeConfig = {
    '1h': 60 * 60 * 1000,      // 1 hour
    '6h': 6 * 60 * 60 * 1000,   // 6 hours
    '24h': 24 * 60 * 60 * 1000, // 24 hours
  };

  // Aggregation intervals for different time ranges
  const aggregationIntervals = {
    '1h': 1000,        // 1 second (no aggregation)
    '6h': 5000,        // 5 seconds
    '24h': 15000,      // 15 seconds
  };

  /**
   * Add a new data point to the time series
   */
  const addDataPoint = useCallback((newData: Omit<MultiSeriesDataPoint, 'timestamp'>) => {
    const timestamp = Date.now();

    setData((prevData) => {
      const newPoint: MultiSeriesDataPoint = {
        timestamp,
        ...newData,
      };

      // Add new point
      const updatedData = [...prevData, newPoint];

      // Prune old data every 60 seconds to avoid excessive processing
      const now = Date.now();
      if (now - lastPruneTime.current > 60000) {
        lastPruneTime.current = now;
        const cutoffTime = now - timeRangeConfig['1h']; // Keep 1h max
        return updatedData.filter((point) => point.timestamp > cutoffTime);
      }

      return updatedData;
    });
  }, [timeRangeConfig]);

  /**
   * Get aggregated data for the selected time range
   */
  const getAggregatedData = useCallback(() => {
    const now = Date.now();
    const cutoffTime = now - timeRangeConfig[timeRange];
    const interval = aggregationIntervals[timeRange];

    // Filter data within time range
    const filteredData = data.filter((point) => point.timestamp > cutoffTime);

    // If no aggregation needed (1h view), return filtered data
    if (interval === 1000) {
      return filteredData;
    }

    // Aggregate data into intervals
    const aggregated: MultiSeriesDataPoint[] = [];
    const buckets = new Map<number, MultiSeriesDataPoint[]>();

    // Group data points into time buckets
    filteredData.forEach((point) => {
      const bucketKey = Math.floor(point.timestamp / interval) * interval;
      if (!buckets.has(bucketKey)) {
        buckets.set(bucketKey, []);
      }
      buckets.get(bucketKey)!.push(point);
    });

    // Calculate averages for each bucket
    buckets.forEach((points, bucketTimestamp) => {
      const keys = Object.keys(points[0]).filter((k) => k !== 'timestamp');
      const aggregatedPoint: MultiSeriesDataPoint = { timestamp: bucketTimestamp };

      keys.forEach((key) => {
        const sum = points.reduce((acc, p) => acc + (p[key] || 0), 0);
        aggregatedPoint[key] = sum / points.length;
      });

      aggregated.push(aggregatedPoint);
    });

    // Sort by timestamp
    return aggregated.sort((a, b) => a.timestamp - b.timestamp);
  }, [data, timeRange, timeRangeConfig, aggregationIntervals]);

  /**
   * Get data for a specific series
   */
  const getSeries = useCallback((seriesKey: string): DataPoint[] => {
    const aggregatedData = getAggregatedData();
    return aggregatedData
      .filter((point) => seriesKey in point)
      .map((point) => ({
        timestamp: point.timestamp,
        value: point[seriesKey],
      }));
  }, [getAggregatedData]);

  /**
   * Clear all historical data
   */
  const clearData = useCallback(() => {
    setData([]);
  }, []);

  /**
   * Get statistics for a series
   */
  const getSeriesStats = useCallback((seriesKey: string) => {
    const series = getSeries(seriesKey);
    if (series.length === 0) {
      return { min: 0, max: 0, avg: 0, current: 0 };
    }

    const values = series.map((p) => p.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const current = series[series.length - 1]?.value || 0;

    return { min, max, avg, current };
  }, [getSeries]);

  return {
    data: getAggregatedData(),
    addDataPoint,
    getSeries,
    getSeriesStats,
    clearData,
    dataPointCount: data.length,
  };
}
