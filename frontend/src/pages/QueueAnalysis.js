import React, { useState, useEffect } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Area,
  AreaChart
} from 'recharts';
import { 
  Shield, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle,
  Search,
  Filter,
  Download
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const QueueAnalysis = () => {
  const [queuePerformance, setQueuePerformance] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedQueue, setSelectedQueue] = useState('all');
  const [timeRange, setTimeRange] = useState('7d');

  useEffect(() => {
    fetchQueueData();
  }, []);

  const fetchQueueData = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/dashboard/queue-performance');
      setQueuePerformance(response.data);
    } catch (error) {
      console.error('Error fetching queue data:', error);
      toast.error('Failed to load queue analysis data');
    } finally {
      setLoading(false);
    }
  };

  const queueColors = {
    'CTI': '#3b82f6',
    'DFIR::incidents': '#ef4444',
    'DFIR::phishing': '#f59e0b',
    'OFFSEC::CVD': '#8b5cf6',
    'OFFSEC::Pentesting': '#10b981',
    'SMS': '#06b6d4',
    'Trash': '#6b7280'
  };

  const getQueueIcon = (queue) => {
    const icons = {
      'CTI': Shield,
      'DFIR::incidents': AlertTriangle,
      'DFIR::phishing': AlertTriangle,
      'OFFSEC::CVD': Shield,
      'OFFSEC::Pentesting': Shield,
      'SMS': CheckCircle,
      'Trash': AlertTriangle
    };
    return icons[queue] || Shield;
  };

  const getQueueDescription = (queue) => {
    const descriptions = {
      'CTI': 'Cyber Threat Intelligence - Analysis of threat indicators and intelligence reports',
      'DFIR::incidents': 'Digital Forensics & Incident Response - Active security incidents requiring investigation',
      'DFIR::phishing': 'Digital Forensics & Incident Response - Phishing attacks and email-based threats',
      'OFFSEC::CVD': 'Offensive Security - Coordinated Vulnerability Disclosure reports',
      'OFFSEC::Pentesting': 'Offensive Security - Penetration testing and security assessments',
      'SMS': 'Security Management Services - General security management and administrative tasks',
      'Trash': 'Irrelevant or spam tickets that should be filtered out'
    };
    return descriptions[queue] || 'No description available';
  };

  // Mock data for trends (in real app, fetch from API)
  const mockTrends = [
    { date: '2024-01-01', CTI: 45, 'DFIR::incidents': 23, 'DFIR::phishing': 67, 'OFFSEC::CVD': 12, SMS: 34, Trash: 8 },
    { date: '2024-01-02', CTI: 52, 'DFIR::incidents': 31, 'DFIR::phishing': 71, 'OFFSEC::CVD': 15, SMS: 28, Trash: 12 },
    { date: '2024-01-03', CTI: 38, 'DFIR::incidents': 19, 'DFIR::phishing': 58, 'OFFSEC::CVD': 18, SMS: 41, Trash: 6 },
    { date: '2024-01-04', CTI: 61, 'DFIR::incidents': 27, 'DFIR::phishing': 73, 'OFFSEC::CVD': 14, SMS: 35, Trash: 9 },
    { date: '2024-01-05', CTI: 49, 'DFIR::incidents': 35, 'DFIR::phishing': 69, 'OFFSEC::CVD': 21, SMS: 29, Trash: 11 },
    { date: '2024-01-06', CTI: 43, 'DFIR::incidents': 22, 'DFIR::phishing': 65, 'OFFSEC::CVD': 16, SMS: 37, Trash: 7 },
    { date: '2024-01-07', CTI: 56, 'DFIR::incidents': 28, 'DFIR::phishing': 75, 'OFFSEC::CVD': 13, SMS: 32, Trash: 10 }
  ];

  const filteredTrends = selectedQueue === 'all' 
    ? mockTrends 
    : mockTrends.map(item => ({
        date: item.date,
        [selectedQueue]: item[selectedQueue]
      }));

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
          <h1 className="text-3xl font-bold text-gray-900">Queue Analysis</h1>
          <p className="text-gray-600 mt-1">Detailed analysis of ticket distribution and queue performance</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={selectedQueue}
            onChange={(e) => setSelectedQueue(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
          >
            <option value="all">All Queues</option>
            {queuePerformance.map(queue => (
              <option key={queue.predicted_queue} value={queue.predicted_queue}>
                {queue.predicted_queue}
              </option>
            ))}
          </select>
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
          <button className="eu-button-secondary flex items-center">
            <Download className="w-4 h-4 mr-2" />
            Export
          </button>
        </div>
      </div>

      {/* Queue Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {queuePerformance.map((queue) => {
          const Icon = getQueueIcon(queue.predicted_queue);
          return (
            <div key={queue.predicted_queue} className="eu-card p-6 fade-in">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <div 
                    className="w-10 h-10 rounded-lg flex items-center justify-center mr-3"
                    style={{ backgroundColor: `${queueColors[queue.predicted_queue]}20` }}
                  >
                    <Icon 
                      className="w-5 h-5" 
                      style={{ color: queueColors[queue.predicted_queue] }}
                    />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">{queue.predicted_queue}</h3>
                    <p className="text-sm text-gray-600">{queue.total_predictions} tickets</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-gray-900">
                    {(queue.avg_confidence * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-600">avg confidence</p>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Auto-routed:</span>
                  <span className="font-medium">{queue.auto_routed_count}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Human review:</span>
                  <span className="font-medium">{queue.human_verify_count}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Error rate:</span>
                  <span className={`font-medium ${
                    queue.error_rate_percent < 5 ? 'text-green-600' : 
                    queue.error_rate_percent < 15 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {queue.error_rate_percent.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Volume Distribution */}
        <div className="eu-card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Volume Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={queuePerformance}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="predicted_queue" 
                angle={-45}
                textAnchor="end"
                height={80}
                fontSize={12}
              />
              <YAxis />
              <Tooltip />
              <Bar dataKey="total_predictions" fill="#003399" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Confidence vs Error Rate */}
        <div className="eu-card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Confidence vs Error Rate</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={queuePerformance}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="predicted_queue" 
                angle={-45}
                textAnchor="end"
                height={80}
                fontSize={12}
              />
              <YAxis />
              <Tooltip />
              <Bar dataKey="avg_confidence" fill="#003399" />
              <Bar dataKey="error_rate_percent" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Trends Chart */}
      <div className="eu-card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Queue Trends Over Time</h3>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={filteredTrends}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            {selectedQueue === 'all' ? (
              Object.keys(queueColors).map((queue, index) => (
                <Area
                  key={queue}
                  type="monotone"
                  dataKey={queue}
                  stackId="1"
                  stroke={queueColors[queue]}
                  fill={queueColors[queue]}
                  fillOpacity={0.6}
                />
              ))
            ) : (
              <Area
                type="monotone"
                dataKey={selectedQueue}
                stackId="1"
                stroke={queueColors[selectedQueue]}
                fill={queueColors[selectedQueue]}
                fillOpacity={0.6}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Queue Information */}
      <div className="eu-card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Queue Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {queuePerformance.map((queue) => (
            <div key={queue.predicted_queue} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center mb-3">
                <div 
                  className="w-6 h-6 rounded-full mr-3"
                  style={{ backgroundColor: queueColors[queue.predicted_queue] }}
                />
                <h4 className="font-semibold text-gray-900">{queue.predicted_queue}</h4>
              </div>
              
              <p className="text-sm text-gray-600 mb-4">
                {getQueueDescription(queue.predicted_queue)}
              </p>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total:</span>
                  <span className="font-medium">{queue.total_predictions}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Auto-routed:</span>
                  <span className="font-medium">{queue.auto_routed_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Human verify:</span>
                  <span className="font-medium">{queue.human_verify_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Manual triage:</span>
                  <span className="font-medium">{queue.manual_triage_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Feedback:</span>
                  <span className="font-medium">{queue.total_feedback}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Corrections:</span>
                  <span className="font-medium">{queue.corrections_count}</span>
                </div>
              </div>
              
              <div className="mt-4 pt-3 border-t border-gray-200">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Avg Confidence:</span>
                  <span className={`font-medium ${
                    queue.avg_confidence > 0.8 ? 'text-green-600' : 
                    queue.avg_confidence > 0.6 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {(queue.avg_confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Error Rate:</span>
                  <span className={`font-medium ${
                    queue.error_rate_percent < 5 ? 'text-green-600' : 
                    queue.error_rate_percent < 15 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {queue.error_rate_percent.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QueueAnalysis;
