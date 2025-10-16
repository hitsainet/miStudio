import { useState, useEffect } from 'react';
import { Cpu, Activity, Thermometer, HardDrive, Settings, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// System Monitoring Tab - Standalone Component
export default function SystemMonitorTab() {
  const [gpuMetrics, setGpuMetrics] = useState({
    utilization: 0,
    memoryUsed: 0,
    memoryTotal: 12288, // 12GB in MB
    temperature: 0,
    maxTemp: 95,
    powerUsage: 0,
    maxPower: 250,
    fanSpeed: 0
  });

  const [diskMetrics, setDiskMetrics] = useState([
    { mount: '/', used: '45GB', total: '128GB', percent: 35 },
    { mount: '/data', used: '234GB', total: '512GB', percent: 46 }
  ]);

  const [hardwareMetrics, setHardwareMetrics] = useState({
    gpuClock: 0,
    maxGpuClock: 1950,
    memoryClock: 0,
    maxMemoryClock: 6800,
    pcieBandwidth: 0,
    maxPcieBandwidth: 16, // GB/s
    encoderUsage: 0,
    decoderUsage: 0
  });

  const [systemMetrics, setSystemMetrics] = useState({
    cpuUsage: 0,
    ramUsed: 0,
    ramTotal: 32768, // MB
    swapUsed: 0,
    swapTotal: 8192, // MB
    networkUp: 0,
    networkDown: 0,
    diskReadSpeed: 0,
    diskWriteSpeed: 0
  });

  const [gpuProcesses, setGpuProcesses] = useState([
    { pid: 12345, name: 'python3', gpuMemory: 4096, cpuUsage: 45.2 },
    { pid: 12346, name: 'torch_train', gpuMemory: 2048, cpuUsage: 32.8 }
  ]);

  const [historicalData, setHistoricalData] = useState([]);
  const [timeRange, setTimeRange] = useState('1h'); // '1h', '6h', '24h'
  const [viewMode, setViewMode] = useState('single'); // 'single' or 'comparison'

  // Multi-GPU metrics for comparison view
  const [multiGpuMetrics, setMultiGpuMetrics] = useState({
    'cuda:0': {
      utilization: 75,
      memoryUsed: 6144,
      memoryTotal: 8192,
      temperature: 68,
      powerUsage: 180,
      maxPower: 250
    },
    'cuda:1': {
      utilization: 62,
      memoryUsed: 18432,
      memoryTotal: 24576,
      temperature: 71,
      powerUsage: 285,
      maxPower: 350
    }
  });

  const [selectedGPU, setSelectedGPU] = useState('cuda:0');
  const [availableGPUs, setAvailableGPUs] = useState([
    { id: 'cuda:0', name: 'NVIDIA Jetson Orin Nano', memory: '8GB' },
    { id: 'cuda:1', name: 'NVIDIA RTX 3090', memory: '24GB' }
  ]);

  // Simulate real-time GPU metrics updates
  useEffect(() => {
    const interval = setInterval(() => {
      const timestamp = new Date().toLocaleTimeString();
      
      setGpuMetrics(prev => {
        const updated = {
          ...prev,
          utilization: Math.max(0, Math.min(100, prev.utilization + (Math.random() - 0.5) * 10)),
          memoryUsed: Math.max(0, Math.min(prev.memoryTotal, prev.memoryUsed + (Math.random() - 0.5) * 200)),
          temperature: Math.max(30, Math.min(prev.maxTemp, prev.temperature + (Math.random() - 0.5) * 2)),
          powerUsage: Math.max(0, Math.min(prev.maxPower, prev.powerUsage + (Math.random() - 0.5) * 10)),
          fanSpeed: Math.max(0, Math.min(100, prev.fanSpeed + (Math.random() - 0.5) * 5))
        };
        
        // Add historical data point using current values
        setHistoricalData(prevHist => {
          const newPoint = {
            time: timestamp,
            gpuUtil: updated.utilization,
            cpuUtil: systemMetrics.cpuUsage,
            gpuMemory: (updated.memoryUsed / updated.memoryTotal) * 100,
            ramUsage: (systemMetrics.ramUsed / systemMetrics.ramTotal) * 100,
            temperature: updated.temperature
          };
          
          const maxPoints = timeRange === '1h' ? 60 : timeRange === '6h' ? 360 : 1440;
          const updatedHist = [...prevHist, newPoint];
          return updatedHist.slice(-maxPoints);
        });
        
        return updated;
      });

      setHardwareMetrics(prev => ({
        ...prev,
        gpuClock: Math.max(300, Math.min(prev.maxGpuClock, prev.gpuClock + (Math.random() - 0.5) * 50)),
        memoryClock: Math.max(300, Math.min(prev.maxMemoryClock, prev.memoryClock + (Math.random() - 0.5) * 100)),
        pcieBandwidth: Math.max(0, Math.min(prev.maxPcieBandwidth, prev.pcieBandwidth + (Math.random() - 0.5) * 2)),
        encoderUsage: Math.max(0, Math.min(100, prev.encoderUsage + (Math.random() - 0.5) * 15)),
        decoderUsage: Math.max(0, Math.min(100, prev.decoderUsage + (Math.random() - 0.5) * 15))
      }));

      setSystemMetrics(prev => ({
        ...prev,
        cpuUsage: Math.max(0, Math.min(100, prev.cpuUsage + (Math.random() - 0.5) * 10)),
        ramUsed: Math.max(0, Math.min(prev.ramTotal, prev.ramUsed + (Math.random() - 0.5) * 512)),
        swapUsed: Math.max(0, Math.min(prev.swapTotal, prev.swapUsed + (Math.random() - 0.5) * 100)),
        networkUp: Math.max(0, prev.networkUp + (Math.random() - 0.5) * 5),
        networkDown: Math.max(0, prev.networkDown + (Math.random() - 0.5) * 10),
        diskReadSpeed: Math.max(0, prev.diskReadSpeed + (Math.random() - 0.5) * 50),
        diskWriteSpeed: Math.max(0, prev.diskWriteSpeed + (Math.random() - 0.5) * 30)
      }));

      // Update multi-GPU metrics
      setMultiGpuMetrics(prev => ({
        'cuda:0': {
          ...prev['cuda:0'],
          utilization: Math.max(0, Math.min(100, prev['cuda:0'].utilization + (Math.random() - 0.5) * 10)),
          memoryUsed: Math.max(0, Math.min(prev['cuda:0'].memoryTotal, prev['cuda:0'].memoryUsed + (Math.random() - 0.5) * 200)),
          temperature: Math.max(30, Math.min(95, prev['cuda:0'].temperature + (Math.random() - 0.5) * 2)),
          powerUsage: Math.max(0, Math.min(prev['cuda:0'].maxPower, prev['cuda:0'].powerUsage + (Math.random() - 0.5) * 10))
        },
        'cuda:1': {
          ...prev['cuda:1'],
          utilization: Math.max(0, Math.min(100, prev['cuda:1'].utilization + (Math.random() - 0.5) * 10)),
          memoryUsed: Math.max(0, Math.min(prev['cuda:1'].memoryTotal, prev['cuda:1'].memoryUsed + (Math.random() - 0.5) * 300)),
          temperature: Math.max(30, Math.min(95, prev['cuda:1'].temperature + (Math.random() - 0.5) * 2)),
          powerUsage: Math.max(0, Math.min(prev['cuda:1'].maxPower, prev['cuda:1'].powerUsage + (Math.random() - 0.5) * 15))
        }
      }));
    }, 1000);

    // Initialize with some values
    setGpuMetrics(prev => ({
      ...prev,
      utilization: 75,
      memoryUsed: 6144,
      temperature: 68,
      powerUsage: 180,
      fanSpeed: 65
    }));

    setHardwareMetrics(prev => ({
      ...prev,
      gpuClock: 1785,
      memoryClock: 6400,
      pcieBandwidth: 8.5,
      encoderUsage: 12,
      decoderUsage: 5
    }));

    setSystemMetrics(prev => ({
      ...prev,
      cpuUsage: 45,
      ramUsed: 18432,
      swapUsed: 1024,
      networkUp: 2.5,
      networkDown: 15.3,
      diskReadSpeed: 320,
      diskWriteSpeed: 180
    }));

    return () => clearInterval(interval);
  }, [timeRange]);

  const getTemperatureColor = (temp) => {
    if (temp < 60) return 'text-emerald-400';
    if (temp < 75) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getUtilizationColor = (util) => {
    if (util < 50) return 'emerald';
    if (util < 80) return 'yellow';
    return 'red';
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold">System Monitoring & Configuration</h2>
          <div className="flex items-center gap-3">
            <div className="text-sm text-slate-400">
              <span className="text-emerald-400">●</span> Live Monitoring
            </div>
            <div className="flex gap-2 bg-slate-800 rounded-lg p-1">
              <button
                onClick={() => setViewMode('single')}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  viewMode === 'single' 
                    ? 'bg-emerald-600 text-white' 
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Single GPU
              </button>
              <button
                onClick={() => setViewMode('comparison')}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  viewMode === 'comparison' 
                    ? 'bg-emerald-600 text-white' 
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Compare GPUs
              </button>
            </div>
            <button className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg flex items-center gap-2 transition-colors">
              <Settings className="w-4 h-4" />
              Settings
            </button>
          </div>
        </div>

        {/* GPU Selection */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">GPU Configuration</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Active GPU Device
              </label>
              <select
                value={selectedGPU}
                onChange={(e) => setSelectedGPU(e.target.value)}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              >
                {availableGPUs.map(gpu => (
                  <option key={gpu.id} value={gpu.id}>
                    {gpu.name} ({gpu.memory})
                  </option>
                ))}
                <option value="cpu">CPU Only</option>
              </select>
            </div>
            <div className="flex items-end">
              <button className="w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded-lg transition-colors">
                Apply Configuration
              </button>
            </div>
          </div>
        </div>

        {/* Historical Trends */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-emerald-400" />
              Historical Trends
            </h3>
            <div className="flex gap-2">
              {['1h', '6h', '24h'].map(range => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-3 py-1 rounded text-sm transition-colors ${
                    timeRange === range
                      ? 'bg-emerald-600 text-white'
                      : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {range}
                </button>
              ))}
            </div>
          </div>

          {historicalData.length > 0 ? (
            <div className="space-y-6">
              {/* GPU & CPU Utilization Chart */}
              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-3">Utilization</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={historicalData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                      dataKey="time" 
                      stroke="#64748b"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                    />
                    <YAxis 
                      stroke="#64748b"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      domain={[0, 100]}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1e293b', 
                        border: '1px solid #334155',
                        borderRadius: '8px'
                      }}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="gpuUtil" 
                      stroke="#a78bfa" 
                      name="GPU %" 
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="cpuUtil" 
                      stroke="#22d3ee" 
                      name="CPU %" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Memory Usage Chart */}
              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-3">Memory Usage</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={historicalData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                      dataKey="time" 
                      stroke="#64748b"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                    />
                    <YAxis 
                      stroke="#64748b"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      domain={[0, 100]}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1e293b', 
                        border: '1px solid #334155',
                        borderRadius: '8px'
                      }}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="gpuMemory" 
                      stroke="#3b82f6" 
                      name="GPU Memory %" 
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="ramUsage" 
                      stroke="#8b5cf6" 
                      name="RAM %" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Temperature Chart */}
              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-3">Temperature</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={historicalData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                      dataKey="time" 
                      stroke="#64748b"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                    />
                    <YAxis 
                      stroke="#64748b"
                      tick={{ fill: '#94a3b8', fontSize: 12 }}
                      domain={[0, 100]}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1e293b', 
                        border: '1px solid #334155',
                        borderRadius: '8px'
                      }}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="temperature" 
                      stroke="#f59e0b" 
                      name="GPU Temp °C" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500">
              <TrendingUp className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>Collecting data... Charts will appear shortly</p>
            </div>
          )}
        </div>

        {/* Multi-GPU Comparison View */}
        {viewMode === 'comparison' && (
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Multi-GPU Comparison</h3>
            <div className="grid grid-cols-2 gap-6">
              {Object.entries(multiGpuMetrics).map(([gpuId, metrics]) => {
                const gpuInfo = availableGPUs.find(g => g.id === gpuId);
                return (
                  <div key={gpuId} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="font-semibold text-emerald-400">{gpuInfo?.name}</h4>
                      <span className="text-xs text-slate-400">{gpuId}</span>
                    </div>

                    <div className="space-y-3">
                      {/* Utilization */}
                      <div>
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span className="text-slate-400">Utilization</span>
                          <span className="text-purple-400 font-mono">
                            {metrics.utilization.toFixed(0)}%
                          </span>
                        </div>
                        <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-500"
                            style={{ width: `${metrics.utilization}%` }}
                          />
                        </div>
                      </div>

                      {/* Memory */}
                      <div>
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span className="text-slate-400">Memory</span>
                          <span className="text-blue-400 font-mono text-xs">
                            {(metrics.memoryUsed / 1024).toFixed(1)} / {(metrics.memoryTotal / 1024).toFixed(1)} GB
                          </span>
                        </div>
                        <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-500"
                            style={{ width: `${(metrics.memoryUsed / metrics.memoryTotal) * 100}%` }}
                          />
                        </div>
                      </div>

                      {/* Temperature */}
                      <div>
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span className="text-slate-400">Temperature</span>
                          <span className="text-orange-400 font-mono">
                            {metrics.temperature.toFixed(0)}°C
                          </span>
                        </div>
                        <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                          <div
                            className={`h-full transition-all duration-500 ${
                              metrics.temperature < 60 ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' :
                              metrics.temperature < 75 ? 'bg-gradient-to-r from-yellow-500 to-yellow-400' :
                              'bg-gradient-to-r from-red-500 to-red-400'
                            }`}
                            style={{ width: `${(metrics.temperature / 95) * 100}%` }}
                          />
                        </div>
                      </div>

                      {/* Power */}
                      <div>
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span className="text-slate-400">Power</span>
                          <span className="text-yellow-400 font-mono">
                            {metrics.powerUsage.toFixed(0)}W
                          </span>
                        </div>
                        <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-yellow-500 to-yellow-400 transition-all duration-500"
                            style={{ width: `${(metrics.powerUsage / metrics.maxPower) * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}


        {/* GPU Metrics Grid */}
        <div className="grid grid-cols-2 gap-6">
          {/* GPU Utilization */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold flex items-center gap-2">
                <Cpu className="w-5 h-5 text-purple-400" />
                GPU Utilization
              </h3>
              <span className={`text-2xl font-bold text-${getUtilizationColor(gpuMetrics.utilization)}-400`}>
                {gpuMetrics.utilization.toFixed(0)}%
              </span>
            </div>
            
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-slate-400">Compute</span>
                  <span className="text-purple-400 font-mono">
                    {gpuMetrics.utilization.toFixed(1)}%
                  </span>
                </div>
                <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-500"
                    style={{ width: `${gpuMetrics.utilization}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-slate-400">Memory</span>
                  <span className="text-blue-400 font-mono">
                    {(gpuMetrics.memoryUsed / 1024).toFixed(1)}GB / {(gpuMetrics.memoryTotal / 1024).toFixed(1)}GB
                  </span>
                </div>
                <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-500"
                    style={{ width: `${(gpuMetrics.memoryUsed / gpuMetrics.memoryTotal) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* GPU Temperature */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold flex items-center gap-2">
                <Thermometer className="w-5 h-5 text-orange-400" />
                Temperature
              </h3>
            </div>
            
            <div className="text-center mb-4">
              <div className={`text-5xl font-bold ${getTemperatureColor(gpuMetrics.temperature)} mb-2`}>
                {gpuMetrics.temperature.toFixed(1)}°C
              </div>
              <div className="text-sm text-slate-400">
                Max: {gpuMetrics.maxTemp}°C
              </div>
            </div>

            <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  gpuMetrics.temperature < 60 ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' :
                  gpuMetrics.temperature < 75 ? 'bg-gradient-to-r from-yellow-500 to-yellow-400' :
                  'bg-gradient-to-r from-red-500 to-red-400'
                }`}
                style={{ width: `${(gpuMetrics.temperature / gpuMetrics.maxTemp) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Additional Metrics */}
        <div className="grid grid-cols-2 gap-6">
          {/* Power Usage */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold flex items-center gap-2">
                <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
                </svg>
                Power Usage
              </h3>
              <span className="text-2xl font-bold text-yellow-400">
                {gpuMetrics.powerUsage.toFixed(0)}W
              </span>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">Current Draw</span>
                <span className="text-slate-300">
                  {gpuMetrics.powerUsage.toFixed(1)}W / {gpuMetrics.maxPower}W
                </span>
              </div>
              <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-yellow-500 to-yellow-400 transition-all duration-500"
                  style={{ width: `${(gpuMetrics.powerUsage / gpuMetrics.maxPower) * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Fan Speed */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold flex items-center gap-2">
                <svg className="w-5 h-5 text-cyan-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M9.504 1.132a1 1 0 01.992 0l1.75 1a1 1 0 11-.992 1.736L10 3.152l-1.254.716a1 1 0 11-.992-1.736l1.75-1zM5.618 4.504a1 1 0 01-.372 1.364L5.016 6l.23.132a1 1 0 11-.992 1.736L4 7.723V8a1 1 0 01-2 0V6a.996.996 0 01.52-.878l1.734-.99a1 1 0 011.364.372zm8.764 0a1 1 0 011.364-.372l1.733.99A1.002 1.002 0 0118 6v2a1 1 0 11-2 0v-.277l-.254.145a1 1 0 11-.992-1.736l.23-.132-.23-.132a1 1 0 01-.372-1.364zm-7 4a1 1 0 011.364-.372L10 8.848l1.254-.716a1 1 0 11.992 1.736L11 10.58V12a1 1 0 11-2 0v-1.42l-1.246-.712a1 1 0 01-.372-1.364zM3 11a1 1 0 011 1v.277l.254.145a1 1 0 11-.992 1.736l-1.75-1a1 1 0 01-.504-.868V11a1 1 0 011-1zm14 0a1 1 0 011 1v1.29a1 1 0 01-.504.868l-1.75 1a1 1 0 11-.992-1.736L15 13.277V12a1 1 0 011-1zm-9.618 5.504a1 1 0 011.364-.372l.254.145V16a1 1 0 112 0v.277l.254-.145a1 1 0 11.992 1.736l-1.75 1a1 1 0 01-.992 0l-1.75-1a1 1 0 01-.372-1.364z" clipRule="evenodd" />
                </svg>
                Fan Speed
              </h3>
              <span className="text-2xl font-bold text-cyan-400">
                {gpuMetrics.fanSpeed.toFixed(0)}%
              </span>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">RPM</span>
                <span className="text-slate-300">
                  {(gpuMetrics.fanSpeed * 25).toFixed(0)} RPM
                </span>
              </div>
              <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 transition-all duration-500"
                  style={{ width: `${gpuMetrics.fanSpeed}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Disk Space */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold flex items-center gap-2">
              <HardDrive className="w-5 h-5 text-emerald-400" />
              Storage
            </h3>
          </div>
          
          <div className="space-y-4">
            {diskMetrics.map(disk => (
              <div key={disk.mount}>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-slate-400 font-mono">{disk.mount}</span>
                  <span className="text-emerald-400 font-mono">
                    {disk.used} / {disk.total} ({disk.percent}%)
                  </span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      disk.percent < 70 ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' :
                      disk.percent < 85 ? 'bg-gradient-to-r from-yellow-500 to-yellow-400' :
                      'bg-gradient-to-r from-red-500 to-red-400'
                    }`}
                    style={{ width: `${disk.percent}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Hardware Metrics */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">Hardware Metrics</h3>
          <div className="grid grid-cols-2 gap-6">
            {/* GPU Clock */}
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate-400">GPU Clock</span>
                <span className="text-purple-400 font-mono">
                  {hardwareMetrics.gpuClock.toFixed(0)} MHz
                </span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-500"
                  style={{ width: `${(hardwareMetrics.gpuClock / hardwareMetrics.maxGpuClock) * 100}%` }}
                />
              </div>
              <div className="text-xs text-slate-500 mt-1">Max: {hardwareMetrics.maxGpuClock} MHz</div>
            </div>

            {/* Memory Clock */}
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate-400">Memory Clock</span>
                <span className="text-blue-400 font-mono">
                  {hardwareMetrics.memoryClock.toFixed(0)} MHz
                </span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-500"
                  style={{ width: `${(hardwareMetrics.memoryClock / hardwareMetrics.maxMemoryClock) * 100}%` }}
                />
              </div>
              <div className="text-xs text-slate-500 mt-1">Max: {hardwareMetrics.maxMemoryClock} MHz</div>
            </div>

            {/* PCIe Bandwidth */}
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate-400">PCIe Bandwidth</span>
                <span className="text-emerald-400 font-mono">
                  {hardwareMetrics.pcieBandwidth.toFixed(1)} GB/s
                </span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500"
                  style={{ width: `${(hardwareMetrics.pcieBandwidth / hardwareMetrics.maxPcieBandwidth) * 100}%` }}
                />
              </div>
              <div className="text-xs text-slate-500 mt-1">Max: {hardwareMetrics.maxPcieBandwidth} GB/s</div>
            </div>

            {/* Encoder/Decoder */}
            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-slate-400">Encoder</span>
                  <span className="text-orange-400 font-mono">
                    {hardwareMetrics.encoderUsage.toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-orange-500 to-orange-400 transition-all duration-500"
                    style={{ width: `${hardwareMetrics.encoderUsage}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-slate-400">Decoder</span>
                  <span className="text-pink-400 font-mono">
                    {hardwareMetrics.decoderUsage.toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-pink-500 to-pink-400 transition-all duration-500"
                    style={{ width: `${hardwareMetrics.decoderUsage}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* System-Level Metrics */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">System Resources</h3>
          <div className="grid grid-cols-2 gap-6">
            {/* CPU Usage */}
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate-400">CPU Utilization</span>
                <span className="text-cyan-400 font-mono">
                  {systemMetrics.cpuUsage.toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 transition-all duration-500"
                  style={{ width: `${systemMetrics.cpuUsage}%` }}
                />
              </div>
            </div>

            {/* RAM Usage */}
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate-400">System RAM</span>
                <span className="text-blue-400 font-mono">
                  {(systemMetrics.ramUsed / 1024).toFixed(1)} / {(systemMetrics.ramTotal / 1024).toFixed(1)} GB
                </span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-500"
                  style={{ width: `${(systemMetrics.ramUsed / systemMetrics.ramTotal) * 100}%` }}
                />
              </div>
            </div>

            {/* Swap Usage */}
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate-400">Swap Memory</span>
                <span className="text-violet-400 font-mono">
                  {(systemMetrics.swapUsed / 1024).toFixed(1)} / {(systemMetrics.swapTotal / 1024).toFixed(1)} GB
                </span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-violet-500 to-violet-400 transition-all duration-500"
                  style={{ width: `${(systemMetrics.swapUsed / systemMetrics.swapTotal) * 100}%` }}
                />
              </div>
            </div>

            {/* Network I/O */}
            <div>
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate-400">Network I/O</span>
                <span className="text-emerald-400 font-mono text-xs">
                  ↑ {systemMetrics.networkUp.toFixed(1)} MB/s ↓ {systemMetrics.networkDown.toFixed(1)} MB/s
                </span>
              </div>
              <div className="space-y-1">
                <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500"
                    style={{ width: `${Math.min(100, (systemMetrics.networkUp / 100) * 100)}%` }}
                  />
                </div>
                <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-500"
                    style={{ width: `${Math.min(100, (systemMetrics.networkDown / 100) * 100)}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Disk I/O */}
            <div className="col-span-2">
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate-400">Disk I/O</span>
                <span className="text-yellow-400 font-mono text-xs">
                  Read: {systemMetrics.diskReadSpeed.toFixed(0)} MB/s | Write: {systemMetrics.diskWriteSpeed.toFixed(0)} MB/s
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <div className="text-xs text-slate-500 mb-1">Read</div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-yellow-500 to-yellow-400 transition-all duration-500"
                      style={{ width: `${Math.min(100, (systemMetrics.diskReadSpeed / 500) * 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 mb-1">Write</div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-orange-500 to-orange-400 transition-all duration-500"
                      style={{ width: `${Math.min(100, (systemMetrics.diskWriteSpeed / 500) * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* GPU Processes */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold flex items-center gap-2">
              <Activity className="w-5 h-5 text-purple-400" />
              GPU Processes
            </h3>
            <span className="text-sm text-slate-400">
              {gpuProcesses.length} active
            </span>
          </div>
          
          {gpuProcesses.length > 0 ? (
            <div className="space-y-2">
              <div className="grid grid-cols-4 gap-4 text-xs text-slate-500 font-medium pb-2 border-b border-slate-700">
                <div>PID</div>
                <div>Process</div>
                <div className="text-right">GPU Memory</div>
                <div className="text-right">CPU %</div>
              </div>
              {gpuProcesses.map(process => (
                <div 
                  key={process.pid}
                  className="grid grid-cols-4 gap-4 items-center bg-slate-800/30 rounded-lg p-3 text-sm"
                >
                  <div className="text-slate-400 font-mono">{process.pid}</div>
                  <div className="text-slate-200 font-medium">{process.name}</div>
                  <div className="text-right text-purple-400 font-mono">
                    {(process.gpuMemory / 1024).toFixed(1)} GB
                  </div>
                  <div className="text-right text-cyan-400 font-mono">
                    {process.cpuUsage.toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-slate-500">
              No active GPU processes
            </div>
          )}
        </div>

        {/* System Info */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">System Information</h3>
          <div className="grid grid-cols-3 gap-6 text-sm">
            <div>
              <div className="text-slate-400 mb-1">Device</div>
              <div className="text-slate-200 font-medium">NVIDIA Jetson Orin Nano</div>
            </div>
            <div>
              <div className="text-slate-400 mb-1">CUDA Version</div>
              <div className="text-slate-200 font-medium">12.2</div>
            </div>
            <div>
              <div className="text-slate-400 mb-1">Driver Version</div>
              <div className="text-slate-200 font-medium">535.104.05</div>
            </div>
            <div>
              <div className="text-slate-400 mb-1">Compute Capability</div>
              <div className="text-slate-200 font-medium">8.7</div>
            </div>
            <div>
              <div className="text-slate-400 mb-1">Total Memory</div>
              <div className="text-slate-200 font-medium">8 GB</div>
            </div>
            <div>
              <div className="text-slate-400 mb-1">Memory Clock</div>
              <div className="text-slate-200 font-medium">6800 MHz</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
