import React, { useState } from 'react';
import { Upload, FileText, Send, Loader, CheckCircle, AlertTriangle } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const TicketAnalyzer = () => {
  const [ticketData, setTicketData] = useState({
    ticket_id: '',
    title: '',
    content: '',
    email_address: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setTicketData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    const fileName = uploadedFile.name.toLowerCase();
    const isValidFile = uploadedFile.type === 'application/json' || 
                       uploadedFile.type === 'text/plain' ||
                       fileName.endsWith('.json') || 
                       fileName.endsWith('.jsonl');
    
    if (uploadedFile && isValidFile) {
      setFile(uploadedFile);
      toast.success('File uploaded successfully');
    } else {
      toast.error('Please upload a JSON or JSONL file');
    }
  };

  const analyzeTicket = async () => {
    if (!ticketData.title || !ticketData.content) {
      toast.error('Please fill in title and content');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post('/api/predict', ticketData);
      setPrediction(response.data);
      toast.success('Ticket analyzed successfully');
    } catch (error) {
      console.error('Error analyzing ticket:', error);
      toast.error('Failed to analyze ticket');
    } finally {
      setLoading(false);
    }
  };

  const analyzeBatch = async () => {
    if (!file) {
      toast.error('Please select a file');
      return;
    }

    setLoading(true);
    try {
      // Read the file content
      const fileContent = await file.text();
      
      // Parse JSONL (each line is a JSON object)
      const lines = fileContent.trim().split('\n');
      const tickets = [];
      
      for (const line of lines) {
        if (line.trim()) {
          try {
            const ticket = JSON.parse(line);
            tickets.push(ticket);
          } catch (parseError) {
            console.error('Error parsing JSON line:', line, parseError);
            toast.error(`Error parsing line: ${line.substring(0, 50)}...`);
            return;
          }
        }
      }
      
      if (tickets.length === 0) {
        toast.error('No valid tickets found in file');
        return;
      }
      
      // Send as JSON data to the new batch endpoint
      const response = await axios.post('/api/predict/batch', {
        tickets: tickets
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      const message = `Successfully processed ${response.data.total_processed || 0} tickets (${response.data.saved || 0} saved, ${response.data.duplicates_skipped || 0} duplicates skipped)`;
      toast.success(`Batch analysis completed: ${message}`);
      setPrediction(null); // Clear single prediction
    } catch (error) {
      console.error('Error analyzing batch:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to analyze batch';
      toast.error(`Batch analysis failed: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  const getQueueColor = (queue) => {
    const colors = {
      'CTI': 'queue-cti',
      'DFIR::incidents': 'queue-dfir-incidents',
      'DFIR::phishing': 'queue-dfir-phishing',
      'OFFSEC::CVD': 'queue-offsec-cvd',
      'OFFSEC::Pentesting': 'queue-offsec-pentesting',
      'SMS': 'queue-sms',
      'Trash': 'queue-trash'
    };
    return colors[queue] || 'queue-trash';
  };

  const getConfidenceClass = (confidence) => {
    if (confidence >= 0.85) return 'confidence-high';
    if (confidence >= 0.65) return 'confidence-medium';
    return 'confidence-low';
  };

  const getRoutingIcon = (routing) => {
    switch (routing) {
      case 'auto_route':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'human_verify':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      default:
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Ticket Analyzer</h1>
        <p className="text-gray-600 mt-1">Analyze individual tickets or upload batch files for classification</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="eu-card p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Single Ticket Analysis</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Ticket ID
              </label>
              <input
                type="text"
                name="ticket_id"
                value={ticketData.ticket_id}
                onChange={handleInputChange}
                placeholder="e.g., TKT-123456"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Title *
              </label>
              <input
                type="text"
                name="title"
                value={ticketData.title}
                onChange={handleInputChange}
                placeholder="Ticket title"
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Content *
              </label>
              <textarea
                name="content"
                value={ticketData.content}
                onChange={handleInputChange}
                placeholder="Ticket content"
                required
                rows={6}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <input
                type="email"
                name="email_address"
                value={ticketData.email_address}
                onChange={handleInputChange}
                placeholder="sender@example.com"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
              />
            </div>

            <button
              onClick={analyzeTicket}
              disabled={loading || !ticketData.title || !ticketData.content}
              className="w-full eu-button-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading ? (
                <Loader className="w-5 h-5 animate-spin mr-2" />
              ) : (
                <Send className="w-5 h-5 mr-2" />
              )}
              Analyze Ticket
            </button>
          </div>
        </div>

        {/* Batch Upload */}
        <div className="eu-card p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Batch Analysis</h2>
          
          <div className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-sm text-gray-600 mb-2">
                Upload JSON or JSONL file for batch processing
              </p>
              <input
                type="file"
                accept=".json,.jsonl,application/json,text/plain"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="eu-button-secondary cursor-pointer inline-flex items-center"
              >
                <FileText className="w-4 h-4 mr-2" />
                Choose File
              </label>
              {file && (
                <p className="text-sm text-green-600 mt-2">
                  Selected: {file.name}
                </p>
              )}
            </div>

            <button
              onClick={analyzeBatch}
              disabled={loading || !file}
              className="w-full eu-button-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading ? (
                <Loader className="w-5 h-5 animate-spin mr-2" />
              ) : (
                <Upload className="w-5 h-5 mr-2" />
              )}
              Process Batch
            </button>
          </div>
        </div>
      </div>

      {/* Prediction Results */}
      {prediction && (
        <div className="eu-card p-6 fade-in">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Analysis Results</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Main Prediction */}
            <div className="space-y-4">
              <div className={`p-4 rounded-lg ${getQueueColor(prediction.predicted_queue)}`}>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold">Predicted Queue</h3>
                  {getRoutingIcon(prediction.routing_decision)}
                </div>
                <p className="text-lg font-bold">{prediction.predicted_queue}</p>
                <p className="text-sm opacity-75">Routing: {prediction.routing_decision.replace('_', ' ')}</p>
              </div>

              <div className={`p-4 rounded-lg ${getConfidenceClass(prediction.confidence_score)}`}>
                <h3 className="font-semibold mb-2">Confidence Score</h3>
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-bold">
                    {(prediction.confidence_score * 100).toFixed(1)}%
                  </span>
                  <div className="w-20 h-2 bg-gray-200 rounded-full">
                    <div 
                      className="h-2 bg-current rounded-full"
                      style={{ width: `${prediction.confidence_score * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold mb-2">Processing Info</h3>
                <p className="text-sm text-gray-600">
                  Processing Time: {prediction.processing_time_ms}ms
                </p>
                <p className="text-sm text-gray-600">
                  Model Version: {prediction.model_version}
                </p>
              </div>
            </div>

            {/* All Probabilities */}
            <div>
              <h3 className="font-semibold mb-4">All Queue Probabilities</h3>
              <div className="space-y-2">
                {Object.entries(prediction.all_probabilities)
                  .sort(([,a], [,b]) => b - a)
                  .map(([queue, probability]) => (
                    <div key={queue} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm font-medium">{queue}</span>
                      <div className="flex items-center">
                        <div className="w-16 h-2 bg-gray-200 rounded-full mr-2">
                          <div 
                            className="h-2 bg-eu-blue-500 rounded-full"
                            style={{ width: `${probability * 100}%` }}
                          />
                        </div>
                        <span className="text-sm text-gray-600 w-12 text-right">
                          {(probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TicketAnalyzer;
