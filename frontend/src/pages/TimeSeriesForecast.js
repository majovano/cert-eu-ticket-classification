import React, { useState, useEffect } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  Calendar,
  BarChart3,
  Brain,
  Play,
  Download,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const TimeSeriesForecast = () => {
  const [historicalData, setHistoricalData] = useState({});
  const [predictions, setPredictions] = useState({});
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [daysAhead, setDaysAhead] = useState(7);
  const [selectedQueue, setSelectedQueue] = useState('all');

  const queueColors = {
    'CTI': '#3b82f6',
    'DFIR::incidents': '#ef4444',
    'DFIR::phishing': '#f59e0b',
    'OFFSEC::CVD': '#8b5cf6',
    'OFFSEC::Pentesting': '#10b981',
    'SMS': '#06b6d4',
    'Trash': '#6b7280'
  };

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [trendsResponse, statusResponse] = await Promise.all([
        axios.get('/api/time-series/trends?days_back=30'),
        axios.get('/api/time-series/models/status')
      ]);
      
      setHistoricalData(trendsResponse.data.trends);
      setModelStatus(statusResponse.data);
    } catch (error) {
      console.error('Error fetching time series data:', error);
      toast.error('Failed to load time series data');
    } finally {
      setLoading(false);
    }
  };

  const trainModels = async () => {
    try {
      setTraining(true);
      const response = await axios.post('/api/time-series/train?days_back=30');
      
      toast.success(`Trained ${response.data.queues.length} models successfully!`);
      
      // Refresh model status
      const statusResponse = await axios.get('/api/time-series/models/status');
      setModelStatus(statusResponse.data);
      
    } catch (error) {
      console.error('Error training models:', error);
      toast.error('Failed to train time series models');
    } finally {
      setTraining(false);
    }
  };

  const makePredictions = async () => {
    try {
      setPredicting(true);
      const response = await axios.get(`/api/time-series/predict?days_ahead=${daysAhead}`);
      
      setPredictions(response.data);
      toast.success(`Generated predictions for ${daysAhead} days ahead`);
      
    } catch (error) {
      console.error('Error making predictions:', error);
      toast.error('Failed to generate predictions');
    } finally {
      setPredicting(false);
    }
  };

  const prepareChartData = () => {
    const allData = [];
    const today = new Date();
    
    // Add historical data
    Object.entries(historicalData).forEach(([queue, data]) => {
      if (selectedQueue === 'all' || selectedQueue === queue) {
        data.forEach(point => {
          allData.push({
            date: point.date,
            queue: queue,
            value: point.ticket_count,
            type: 'historical',
            moving_average: point.moving_average_7d
          });
        });
      }
    });
    
    // Add predictions
    if (predictions.predictions) {
      Object.entries(predictions.predictions).forEach(([queue, data]) => {
        if (selectedQueue === 'all' || selectedQueue === queue) {
          data.forEach(point => {
            allData.push({
              date: point.date,
              queue: queue,
              value: point.predicted_tickets,
              type: 'prediction',
              confidence: point.confidence
            });
          });
        }
      });
    }
    
    // Group by date and queue
    const groupedData = {};
    allData.forEach(point => {
      const key = point.date;
      if (!groupedData[key]) {
        groupedData[key] = { date: point.date };
      }
      groupedData[key][`${point.queue}_${point.type}`] = point.value;
      groupedData[key][`${point.queue}_confidence`] = point.confidence || 1;
    });
    
    return Object.values(groupedData).sort((a, b) => new Date(a.date) - new Date(b.date));
  };

  const exportPredictions = () => {
    if (!predictions.predictions) return;
    
    const csvData = [];
    Object.entries(predictions.predictions).forEach(([queue, data]) => {
      data.forEach(point => {
        csvData.push({
          queue,
          date: point.date,
          predicted_tickets: point.predicted_tickets,
          confidence: point.confidence
        });
      });
    });
    
    const csv = [
      'Queue,Date,Predicted Tickets,Confidence',
      ...csvData.map(row => `${row.queue},${row.date},${row.predicted_tickets},${row.confidence}`)
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ticket_predictions_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
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
          <h1 className="text-3xl font-bold text-gray-900">Time Series Forecasting</h1>
          <p className="text-gray-600 mt-1">Predict future ticket volumes using historical patterns</p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={trainModels}
            disabled={training}
            className="eu-button flex items-center space-x-2"
          >
            <Brain className="w-4 h-4" />
            <span>{training ? 'Training...' : 'Train Models'}</span>
          </button>
          <button
            onClick={makePredictions}
            disabled={predicting || !modelStatus?.models_loaded}
            className="eu-button-secondary flex items-center space-x-2"
          >
            <Play className="w-4 h-4" />
            <span>{predicting ? 'Predicting...' : 'Generate Predictions'}</span>
          </button>
        </div>
      </div>

      {/* Model Status */}
      <div className="eu-card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Status</h3>
        {modelStatus?.models_loaded ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center p-4 bg-green-50 rounded-lg">
              <CheckCircle className="w-8 h-8 text-green-500 mr-3" />
              <div>
                <p className="font-medium text-gray-900">Models Ready</p>
                <p className="text-sm text-gray-600">{modelStatus.total_models} queues trained</p>
              </div>
            </div>
            <div className="flex items-center p-4 bg-blue-50 rounded-lg">
              <BarChart3 className="w-8 h-8 text-blue-500 mr-3" />
              <div>
                <p className="font-medium text-gray-900">Avg Performance</p>
                <p className="text-sm text-gray-600">MAE: {Object.values(modelStatus.model_details).reduce((acc, model) => acc + model.mae, 0) / Object.keys(modelStatus.model_details).length}</p>
              </div>
            </div>
            <div className="flex items-center p-4 bg-purple-50 rounded-lg">
              <Brain className="w-8 h-8 text-purple-500 mr-3" />
              <div>
                <p className="font-medium text-gray-900">Confidence</p>
                <p className="text-sm text-gray-600">{Object.values(modelStatus.model_details).reduce((acc, model) => acc + model.confidence, 0) / Object.keys(modelStatus.model_details).length}</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center p-4 bg-yellow-50 rounded-lg">
            <AlertTriangle className="w-8 h-8 text-yellow-500 mr-3" />
            <div>
              <p className="font-medium text-gray-900">No Models Trained</p>
              <p className="text-sm text-gray-600">Click "Train Models" to start forecasting</p>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="eu-card p-6">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Prediction Period
            </label>
            <select
              value={daysAhead}
              onChange={(e) => setDaysAhead(Number(e.target.value))}
              className="eu-input"
            >
              <option value={3}>3 days ahead</option>
              <option value={7}>7 days ahead</option>
              <option value={14}>14 days ahead</option>
              <option value={30}>30 days ahead</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Queue Filter
            </label>
            <select
              value={selectedQueue}
              onChange={(e) => setSelectedQueue(e.target.value)}
              className="eu-input"
            >
              <option value="all">All Queues</option>
              {Object.keys(historicalData).map(queue => (
                <option key={queue} value={queue}>{queue}</option>
              ))}
            </select>
          </div>
          {predictions.predictions && (
            <div className="ml-auto">
              <button
                onClick={exportPredictions}
                className="eu-button-secondary flex items-center space-x-2"
              >
                <Download className="w-4 h-4" />
                <span>Export Predictions</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Time Series Chart */}
      <div className="eu-card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Historical Trends & Predictions
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={prepareChartData()}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date" 
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis />
            <Tooltip 
              labelFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <Legend />
            {Object.keys(historicalData).map(queue => (
              <Area
                key={`${queue}_historical`}
                type="monotone"
                dataKey={`${queue}_historical`}
                stackId="1"
                stroke={queueColors[queue]}
                fill={queueColors[queue]}
                fillOpacity={0.6}
                name={`${queue} (Historical)`}
              />
            ))}
            {Object.keys(predictions.predictions || {}).map(queue => (
              <Area
                key={`${queue}_prediction`}
                type="monotone"
                dataKey={`${queue}_prediction`}
                stackId="2"
                stroke={queueColors[queue]}
                fill={queueColors[queue]}
                fillOpacity={0.3}
                strokeDasharray="5 5"
                name={`${queue} (Predicted)`}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Prediction Summary */}
      {predictions.summary && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Summary Stats */}
          <div className="eu-card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Summary</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Total Predicted Tickets:</span>
                <span className="font-semibold text-lg">{predictions.summary.total_predicted_tickets}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Prediction Period:</span>
                <span className="font-semibold">{predictions.prediction_period}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Generated:</span>
                <span className="font-semibold">{new Date(predictions.generated_at).toLocaleString()}</span>
              </div>
            </div>
          </div>

          {/* Queue Breakdown */}
          <div className="eu-card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Queue Breakdown</h3>
            <div className="space-y-3">
              {Object.entries(predictions.summary.queue_summary).map(([queue, stats]) => (
                <div key={queue} className="flex justify-between items-center">
                  <div className="flex items-center">
                    <div 
                      className="w-3 h-3 rounded-full mr-3"
                      style={{ backgroundColor: queueColors[queue] }}
                    />
                    <span className="text-gray-700">{queue}</span>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">{stats.total_predicted}</div>
                    <div className="text-sm text-gray-500">avg: {stats.average_daily}/day</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Peak Days */}
      {predictions.summary?.peak_days && (
        <div className="eu-card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Peak Prediction Days</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(predictions.summary.peak_days).map(([queue, peak]) => (
              <div key={queue} className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center mb-2">
                  <div 
                    className="w-3 h-3 rounded-full mr-2"
                    style={{ backgroundColor: queueColors[queue] }}
                  />
                  <span className="font-medium">{queue}</span>
                </div>
                <div className="text-sm text-gray-600">
                  Peak: {new Date(peak.date).toLocaleDateString()}
                </div>
                <div className="text-lg font-semibold">
                  {peak.tickets} tickets
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TimeSeriesForecast;
