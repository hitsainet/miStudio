/**
 * LoadingSkeleton Component
 *
 * Skeleton loading state for System Monitor
 */

export function LoadingSkeleton() {
  return (
    <div className="min-h-screen bg-slate-950">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-6 animate-pulse">
        {/* Header Skeleton */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-slate-800 rounded"></div>
            <div>
              <div className="h-8 w-48 bg-slate-800 rounded mb-2"></div>
              <div className="h-4 w-64 bg-slate-800 rounded"></div>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="h-6 w-16 bg-slate-800 rounded"></div>
            <div className="w-10 h-10 bg-slate-800 rounded-lg"></div>
          </div>
        </div>

        {/* GPU Info Skeleton */}
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-800">
          <div className="h-6 w-40 bg-slate-800 rounded mb-4"></div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i}>
                <div className="h-4 w-20 bg-slate-800 rounded mb-2"></div>
                <div className="h-5 w-32 bg-slate-800 rounded"></div>
              </div>
            ))}
          </div>
        </div>

        {/* Metrics Grid Skeleton */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-slate-900 rounded-lg p-4 border border-slate-800">
              <div className="h-4 w-24 bg-slate-800 rounded mb-2"></div>
              <div className="h-10 w-20 bg-slate-800 rounded mb-3"></div>
              <div className="w-full h-2 bg-slate-800 rounded-full"></div>
            </div>
          ))}
        </div>

        {/* Charts Skeleton */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="h-6 w-40 bg-slate-800 rounded"></div>
            <div className="h-10 w-40 bg-slate-800 rounded-lg"></div>
          </div>
          <div className="grid grid-cols-1 gap-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-slate-900 rounded-lg p-6 border border-slate-800">
                <div className="h-64 bg-slate-800 rounded"></div>
              </div>
            ))}
          </div>
        </div>

        {/* System Resources Skeleton */}
        <div className="space-y-4">
          <div className="h-6 w-40 bg-slate-800 rounded"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="bg-slate-900 rounded-lg p-4 border border-slate-800">
                <div className="h-4 w-24 bg-slate-800 rounded mb-2"></div>
                <div className="h-10 w-20 bg-slate-800 rounded mb-3"></div>
                <div className="w-full h-2 bg-slate-800 rounded-full"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
