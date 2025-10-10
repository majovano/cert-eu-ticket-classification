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
  Line
} from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle,
  Clock,
  Users,
  FileText,
  Shield
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const [dashboardStats, setDashboardStats] = useState(null);
  const [queuePerformance, setQueuePerformance] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [statsResponse, queueResponse] = await Promise.all([
        axios.get('/api/dashboard/stats'),
        axios.get('/api/dashboard/queue-performance')
      ]);
      
      setDashboardStats(statsResponse.data);
      setQueuePerformance(queueResponse.data);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    {
      title: 'Total Tickets',
      value: dashboardStats?.total_tickets || 0,
      icon: FileText,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      change: '+12%',
      changeType: 'positive'
    },
    {
      title: 'Auto-Routed',
      value: dashboardStats?.auto_routed || 0,
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      change: '+8%',
      changeType: 'positive'
    },
    {
      title: 'Human Review',
      value: dashboardStats?.human_verify || 0,
      icon: Users,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-50',
      change: '-3%',
      changeType: 'negative'
    },
    {
      title: 'Manual Triage',
      value: dashboardStats?.manual_triage || 0,
      icon: AlertTriangle,
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      change: '-5%',
      changeType: 'negative'
    }
  ];

  const routingData = [
    { name: 'Auto-Route', value: dashboardStats?.auto_routed || 0, color: '#22c55e' },
    { name: 'Human Verify', value: dashboardStats?.human_verify || 0, color: '#f59e0b' },
    { name: 'Manual Triage', value: dashboardStats?.manual_triage || 0, color: '#ef4444' }
  ];

  const queueColors = {
    'CTI': '#3b82f6',
    'DFIR::incidents': '#ef4444',
    'DFIR::phishing': '#f59e0b',
    'OFFSEC::CVD': '#8b5cf6',
    'OFFSEC::Pentesting': '#10b981',
    'SMS': '#06b6d4',
    'Trash': '#6b7280'
  };

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
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">CERT-EU Ticket Classification Overview</p>
        </div>
        <div className="text-sm text-gray-500">
          Last updated: {new Date().toLocaleString()}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div key={index} className="eu-card p-6 fade-in">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                  <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value.toLocaleString()}</p>
                  <div className="flex items-center mt-2">
                    {stat.changeType === 'positive' ? (
                      <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
                    )}
                    <span className={`text-sm font-medium ${
                      stat.changeType === 'positive' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {stat.change}
                    </span>
                    <span className="text-sm text-gray-500 ml-1">from last week</span>
                  </div>
                </div>
                <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                  <Icon className={`w-6 h-6 ${stat.color}`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Queue Distribution */}
        <div className="eu-card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Queue Distribution</h3>
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

        {/* Routing Decisions */}
        <div className="eu-card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Routing Decisions</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={routingData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
              >
                {routingData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex justify-center space-x-6 mt-4">
            {routingData.map((item, index) => (
              <div key={index} className="flex items-center">
                <div 
                  className="w-3 h-3 rounded-full mr-2" 
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-sm text-gray-600">{item.name}: {item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Queue Performance Table */}
      <div className="eu-card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Queue Performance</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Queue
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Predictions
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Avg Confidence
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Auto-Routed
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Error Rate
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {queuePerformance.map((queue, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div 
                        className="w-3 h-3 rounded-full mr-3"
                        style={{ backgroundColor: queueColors[queue.predicted_queue] || '#6b7280' }}
                      />
                      <span className="text-sm font-medium text-gray-900">
                        {queue.predicted_queue}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {queue.total_predictions}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="text-sm text-gray-900">
                        {(queue.avg_confidence * 100).toFixed(1)}%
                      </span>
                      <div className={`ml-2 w-16 h-2 rounded-full ${
                        queue.avg_confidence > 0.8 ? 'bg-green-200' : 
                        queue.avg_confidence > 0.6 ? 'bg-yellow-200' : 'bg-red-200'
                      }`}>
                        <div 
                          className={`h-2 rounded-full ${
                            queue.avg_confidence > 0.8 ? 'bg-green-500' : 
                            queue.avg_confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${queue.avg_confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {queue.auto_routed_count}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-sm font-medium ${
                      queue.error_rate_percent < 5 ? 'text-green-600' : 
                      queue.error_rate_percent < 15 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {queue.error_rate_percent.toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* System Status */}
      <div className="eu-card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center p-4 bg-green-50 rounded-lg">
            <CheckCircle className="w-8 h-8 text-green-500 mr-3" />
            <div>
              <p className="font-medium text-gray-900">Model Status</p>
              <p className="text-sm text-gray-600">Online & Ready</p>
            </div>
          </div>
          <div className="flex items-center p-4 bg-blue-50 rounded-lg">
            <Clock className="w-8 h-8 text-blue-500 mr-3" />
            <div>
              <p className="font-medium text-gray-900">Avg Processing Time</p>
              <p className="text-sm text-gray-600">245ms</p>
            </div>
          </div>
          <div className="flex items-center p-4 bg-yellow-50 rounded-lg">
            <Shield className="w-8 h-8 text-yellow-500 mr-3" />
            <div>
              <p className="font-medium text-gray-900">Security Level</p>
              <p className="text-sm text-gray-600">High</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
