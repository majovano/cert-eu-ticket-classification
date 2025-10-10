import React, { useState, useEffect } from 'react';
import { 
  FileText, 
  Download, 
  Calendar,
  Filter,
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart,
  Activity
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const Reports = () => {
  const [dashboardStats, setDashboardStats] = useState(null);
  const [queuePerformance, setQueuePerformance] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState('overview');
  const [dateRange, setDateRange] = useState('7d');

  useEffect(() => {
    fetchReportData();
  }, [dateRange]);

  const fetchReportData = async () => {
    try {
      setLoading(true);
      const [statsResponse, queueResponse] = await Promise.all([
        axios.get('/api/dashboard/stats'),
        axios.get('/api/dashboard/queue-performance')
      ]);
      
      setDashboardStats(statsResponse.data);
      setQueuePerformance(queueResponse.data);
    } catch (error) {
      console.error('Error fetching report data:', error);
      toast.error('Failed to load report data');
    } finally {
      setLoading(false);
    }
  };

  const generateReport = () => {
    const reportData = {
      generated_at: new Date().toISOString(),
      date_range: dateRange,
      stats: dashboardStats,
      queue_performance: queuePerformance,
      selected_report: selectedReport
    };

    const dataStr = JSON.stringify(reportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `cert-eu-report-${selectedReport}-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    toast.success('Report downloaded successfully');
  };

  const exportToCSV = () => {
    const csvData = queuePerformance.map(queue => ({
      'Queue': queue.predicted_queue,
      'Total Predictions': queue.total_predictions,
      'Average Confidence': (queue.avg_confidence * 100).toFixed(2) + '%',
      'Auto-Routed': queue.auto_routed_count,
      'Human Verify': queue.human_verify_count,
      'Manual Triage': queue.manual_triage_count,
      'Total Feedback': queue.total_feedback,
      'Corrections': queue.corrections_count,
      'Error Rate': queue.error_rate_percent.toFixed(2) + '%'
    }));

    const headers = Object.keys(csvData[0]).join(',');
    const rows = csvData.map(row => Object.values(row).join(','));
    const csvContent = [headers, ...rows].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `cert-eu-queue-performance-${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    
    toast.success('CSV exported successfully');
  };

  const reportTypes = [
    {
      id: 'overview',
      name: 'Overview Report',
      description: 'High-level statistics and key performance indicators',
      icon: BarChart3
    },
    {
      id: 'queue-performance',
      name: 'Queue Performance',
      description: 'Detailed analysis of each queue\'s performance metrics',
      icon: PieChart
    },
    {
      id: 'confidence-analysis',
      name: 'Confidence Analysis',
      description: 'Analysis of prediction confidence and routing decisions',
      icon: Activity
    },
    {
      id: 'feedback-report',
      name: 'Feedback Report',
      description: 'Human feedback analysis and model improvement insights',
      icon: FileText
    }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-eu-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Reports</h1>
          <p className="text-gray-600 mt-1">Generate and export comprehensive analysis reports</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
          <button
            onClick={generateReport}
            className="eu-button-primary flex items-center"
          >
            <Download className="w-4 h-4 mr-2" />
            Generate Report
          </button>
        </div>
      </div>

      {/* Report Type Selection */}
      <div className="eu-card p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Select Report Type</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {reportTypes.map((report) => {
            const Icon = report.icon;
            return (
              <button
                key={report.id}
                onClick={() => setSelectedReport(report.id)}
                className={`p-4 rounded-lg border-2 text-left transition-colors ${
                  selectedReport === report.id
                    ? 'border-eu-blue-500 bg-eu-blue-50'
                    : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                }`}
              >
                <Icon className={`w-6 h-6 mb-2 ${
                  selectedReport === report.id ? 'text-eu-blue-600' : 'text-gray-400'
                }`} />
                <h3 className="font-semibold text-gray-900">{report.name}</h3>
                <p className="text-sm text-gray-600 mt-1">{report.description}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Report Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Report Preview */}
        <div className="lg:col-span-2">
          <div className="eu-card p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-gray-900">
                {reportTypes.find(r => r.id === selectedReport)?.name}
              </h2>
              <div className="flex items-center space-x-2">
                <Calendar className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-600">
                  Generated: {new Date().toLocaleDateString()}
                </span>
              </div>
            </div>

            {selectedReport === 'overview' && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <p className="text-sm text-blue-600 font-medium">Total Tickets</p>
                    <p className="text-2xl font-bold text-blue-900">
                      {dashboardStats?.total_tickets || 0}
                    </p>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg">
                    <p className="text-sm text-green-600 font-medium">Auto-Routed</p>
                    <p className="text-2xl font-bold text-green-900">
                      {dashboardStats?.auto_routed || 0}
                    </p>
                  </div>
                  <div className="bg-yellow-50 p-4 rounded-lg">
                    <p className="text-sm text-yellow-600 font-medium">Human Review</p>
                    <p className="text-2xl font-bold text-yellow-900">
                      {dashboardStats?.human_verify || 0}
                    </p>
                  </div>
                  <div className="bg-red-50 p-4 rounded-lg">
                    <p className="text-sm text-red-600 font-medium">Manual Triage</p>
                    <p className="text-2xl font-bold text-red-900">
                      {dashboardStats?.manual_triage || 0}
                    </p>
                  </div>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-semibold text-gray-900 mb-2">Key Metrics</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Average Confidence:</span>
                      <span className="ml-2 font-medium">
                        {((dashboardStats?.avg_confidence || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Total Feedback:</span>
                      <span className="ml-2 font-medium">{dashboardStats?.total_feedback || 0}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Corrections Needed:</span>
                      <span className="ml-2 font-medium">{dashboardStats?.corrections_needed || 0}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Auto-Route Rate:</span>
                      <span className="ml-2 font-medium">
                        {dashboardStats?.total_predictions > 0 
                          ? ((dashboardStats.auto_routed / dashboardStats.total_predictions) * 100).toFixed(1)
                          : 0
                        }%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedReport === 'queue-performance' && (
              <div className="space-y-4">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Queue</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Predictions</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Auto-Route</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Error Rate</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {queuePerformance.map((queue, index) => (
                        <tr key={index}>
                          <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                            {queue.predicted_queue}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                            {queue.total_predictions}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                            {(queue.avg_confidence * 100).toFixed(1)}%
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                            {queue.auto_routed_count}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                            {queue.error_rate_percent.toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {selectedReport === 'confidence-analysis' && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-green-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-green-900">High Confidence</h3>
                    <p className="text-2xl font-bold text-green-900">
                      {queuePerformance.reduce((sum, q) => sum + q.auto_routed_count, 0)}
                    </p>
                    <p className="text-sm text-green-600">Auto-routed tickets</p>
                  </div>
                  <div className="bg-yellow-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-yellow-900">Medium Confidence</h3>
                    <p className="text-2xl font-bold text-yellow-900">
                      {queuePerformance.reduce((sum, q) => sum + q.human_verify_count, 0)}
                    </p>
                    <p className="text-sm text-yellow-600">Human verification needed</p>
                  </div>
                  <div className="bg-red-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-red-900">Low Confidence</h3>
                    <p className="text-2xl font-bold text-red-900">
                      {queuePerformance.reduce((sum, q) => sum + q.manual_triage_count, 0)}
                    </p>
                    <p className="text-sm text-red-600">Manual triage required</p>
                  </div>
                </div>
              </div>
            )}

            {selectedReport === 'feedback-report' && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-blue-900">Total Feedback</h3>
                    <p className="text-2xl font-bold text-blue-900">
                      {queuePerformance.reduce((sum, q) => sum + q.total_feedback, 0)}
                    </p>
                    <p className="text-sm text-blue-600">Reviews completed</p>
                  </div>
                  <div className="bg-orange-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-orange-900">Corrections</h3>
                    <p className="text-2xl font-bold text-orange-900">
                      {queuePerformance.reduce((sum, q) => sum + q.corrections_count, 0)}
                    </p>
                    <p className="text-sm text-orange-600">Predictions corrected</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Export Options */}
        <div className="space-y-6">
          <div className="eu-card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Export Options</h3>
            <div className="space-y-3">
              <button
                onClick={generateReport}
                className="w-full eu-button-primary flex items-center justify-center"
              >
                <FileText className="w-4 h-4 mr-2" />
                Download JSON Report
              </button>
              <button
                onClick={exportToCSV}
                className="w-full eu-button-secondary flex items-center justify-center"
              >
                <Download className="w-4 h-4 mr-2" />
                Export CSV Data
              </button>
            </div>
          </div>

          <div className="eu-card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Report Settings</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Date Range
                </label>
                <select
                  value={dateRange}
                  onChange={(e) => setDateRange(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
                >
                  <option value="7d">Last 7 days</option>
                  <option value="30d">Last 30 days</option>
                  <option value="90d">Last 90 days</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Report Type
                </label>
                <select
                  value={selectedReport}
                  onChange={(e) => setSelectedReport(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
                >
                  {reportTypes.map(report => (
                    <option key={report.id} value={report.id}>
                      {report.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className="eu-card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Stats</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Tickets:</span>
                <span className="font-medium">{dashboardStats?.total_tickets || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Auto-Route Rate:</span>
                <span className="font-medium">
                  {dashboardStats?.total_predictions > 0 
                    ? ((dashboardStats.auto_routed / dashboardStats.total_predictions) * 100).toFixed(1)
                    : 0
                  }%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Confidence:</span>
                <span className="font-medium">
                  {((dashboardStats?.avg_confidence || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Feedback Rate:</span>
                <span className="font-medium">
                  {dashboardStats?.total_predictions > 0 
                    ? ((dashboardStats.total_feedback / dashboardStats.total_predictions) * 100).toFixed(1)
                    : 0
                  }%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Reports;
