import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Filter,
  Search,
  ThumbsUp,
  ThumbsDown,
  Save,
  Loader
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const HumanReview = () => {
  const [lowConfidenceTickets, setLowConfidenceTickets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filteredTickets, setFilteredTickets] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedQueue, setSelectedQueue] = useState('all');
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [feedback, setFeedback] = useState({
    corrected_queue: '',
    feedback_notes: '',
    keywords_highlighted: [],
    difficulty_score: 3,
    is_correct: null
  });
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetchLowConfidenceTickets();
  }, []);

  useEffect(() => {
    filterTickets();
  }, [lowConfidenceTickets, searchTerm, selectedQueue]);

  const fetchLowConfidenceTickets = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/tickets/low-confidence?limit=100&threshold=0.75');
      setLowConfidenceTickets(response.data.tickets);
    } catch (error) {
      console.error('Error fetching low confidence tickets:', error);
      toast.error('Failed to load tickets for review');
    } finally {
      setLoading(false);
    }
  };

  const filterTickets = () => {
    let filtered = lowConfidenceTickets;

    if (searchTerm) {
      filtered = filtered.filter(ticket => 
        ticket.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        ticket.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
        ticket.ticket_id.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (selectedQueue !== 'all') {
      filtered = filtered.filter(ticket => ticket.predicted_queue === selectedQueue);
    }

    setFilteredTickets(filtered);
  };

  const handleTicketSelect = (ticket) => {
    setSelectedTicket(ticket);
    setFeedback({
      corrected_queue: ticket.predicted_queue,
      feedback_notes: '',
      keywords_highlighted: [],
      difficulty_score: 3,
      is_correct: null
    });
  };

  const handleKeywordHighlight = (keyword) => {
    const keywords = feedback.keywords_highlighted;
    if (keywords.includes(keyword)) {
      setFeedback({
        ...feedback,
        keywords_highlighted: keywords.filter(k => k !== keyword)
      });
    } else {
      setFeedback({
        ...feedback,
        keywords_highlighted: [...keywords, keyword]
      });
    }
  };

  const submitFeedback = async () => {
    if (!selectedTicket || feedback.is_correct === null) {
      toast.error('Please provide feedback for the prediction');
      return;
    }

    setSubmitting(true);
    try {
      await axios.post('/api/feedback', {
        prediction_id: selectedTicket.prediction_id,
        reviewer_id: 'user-123', // In real app, get from auth
        corrected_queue: feedback.corrected_queue,
        feedback_notes: feedback.feedback_notes,
        keywords_highlighted: feedback.keywords_highlighted,
        difficulty_score: feedback.difficulty_score,
        is_correct: feedback.is_correct
      });

      toast.success('Feedback submitted successfully');
      
      // Remove reviewed ticket from list
      setLowConfidenceTickets(prev => 
        prev.filter(ticket => ticket.prediction_id !== selectedTicket.prediction_id)
      );
      setSelectedTicket(null);
      
    } catch (error) {
      console.error('Error submitting feedback:', error);
      toast.error('Failed to submit feedback');
    } finally {
      setSubmitting(false);
    }
  };

  const getConfidenceClass = (confidence) => {
    if (confidence >= 0.7) return 'confidence-medium';
    return 'confidence-low';
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

  const extractKeywords = (text) => {
    // Simple keyword extraction - in real app, use NLP
    const keywords = [
      'phishing', 'malware', 'virus', 'attack', 'breach', 'vulnerability',
      'exploit', 'incident', 'threat', 'fraud', 'scam', 'password', 'login',
      'authentication', 'firewall', 'antivirus', 'encryption', 'certificate',
      'ssl', 'tls', 'vpn', 'backup', 'patch', 'admin', 'privilege', 'access'
    ];
    
    return keywords.filter(keyword => 
      text.toLowerCase().includes(keyword.toLowerCase())
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader className="w-8 h-8 animate-spin text-eu-blue-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Human Review</h1>
        <p className="text-gray-600 mt-1">Review low-confidence predictions and provide feedback</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="eu-card p-4">
          <div className="flex items-center">
            <AlertTriangle className="w-8 h-8 text-yellow-500 mr-3" />
            <div>
              <p className="text-2xl font-bold text-gray-900">{lowConfidenceTickets.length}</p>
              <p className="text-sm text-gray-600">Pending Review</p>
            </div>
          </div>
        </div>
        <div className="eu-card p-4">
          <div className="flex items-center">
            <Clock className="w-8 h-8 text-blue-500 mr-3" />
            <div>
              <p className="text-2xl font-bold text-gray-900">
                {lowConfidenceTickets.filter(t => t.confidence_score < 0.65).length}
              </p>
              <p className="text-sm text-gray-600">Manual Triage</p>
            </div>
          </div>
        </div>
        <div className="eu-card p-4">
          <div className="flex items-center">
            <CheckCircle className="w-8 h-8 text-green-500 mr-3" />
            <div>
              <p className="text-2xl font-bold text-gray-900">
                {lowConfidenceTickets.filter(t => t.confidence_score >= 0.65).length}
              </p>
              <p className="text-sm text-gray-600">Human Verify</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Ticket List */}
        <div className="lg:col-span-1">
          <div className="eu-card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Tickets for Review</h2>
            
            {/* Filters */}
            <div className="space-y-3 mb-4">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-3 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search tickets..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
                />
              </div>
              
              <select
                value={selectedQueue}
                onChange={(e) => setSelectedQueue(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
              >
                <option value="all">All Queues</option>
                <option value="CTI">CTI</option>
                <option value="DFIR::incidents">DFIR::incidents</option>
                <option value="DFIR::phishing">DFIR::phishing</option>
                <option value="OFFSEC::CVD">OFFSEC::CVD</option>
                <option value="OFFSEC::Pentesting">OFFSEC::Pentesting</option>
                <option value="SMS">SMS</option>
                <option value="Trash">Trash</option>
              </select>
            </div>

            {/* Ticket List */}
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {filteredTickets.map((ticket) => (
                <div
                  key={ticket.prediction_id}
                  onClick={() => handleTicketSelect(ticket)}
                  className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                    selectedTicket?.prediction_id === ticket.prediction_id
                      ? 'border-eu-blue-500 bg-eu-blue-50'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <span className="text-sm font-medium text-gray-900">
                      {ticket.ticket_id}
                    </span>
                    <span className={`text-xs px-2 py-1 rounded-full ${getConfidenceClass(ticket.confidence_score)}`}>
                      {(ticket.confidence_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 line-clamp-2">
                    {ticket.title}
                  </p>
                  <div className="flex items-center mt-2">
                    <div className={`px-2 py-1 rounded text-xs ${getQueueColor(ticket.predicted_queue)}`}>
                      {ticket.predicted_queue}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Ticket Review */}
        <div className="lg:col-span-2">
          {selectedTicket ? (
            <div className="eu-card p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-gray-900">
                  Review Ticket: {selectedTicket.ticket_id}
                </h2>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">Confidence:</span>
                  <span className={`text-sm font-medium ${getConfidenceClass(selectedTicket.confidence_score)}`}>
                    {(selectedTicket.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Ticket Content */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 mb-2">Title</h3>
                <p className="text-gray-700 mb-4">{selectedTicket.title}</p>
                
                <h3 className="font-semibold text-gray-900 mb-2">Content</h3>
                <div className="bg-gray-50 p-4 rounded-lg max-h-48 overflow-y-auto">
                  <p className="text-gray-700 whitespace-pre-wrap">{selectedTicket.content}</p>
                </div>
              </div>

              {/* Keywords */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 mb-2">Highlighted Keywords</h3>
                <div className="flex flex-wrap gap-2">
                  {extractKeywords(selectedTicket.title + ' ' + selectedTicket.content).map((keyword) => (
                    <button
                      key={keyword}
                      onClick={() => handleKeywordHighlight(keyword)}
                      className={`px-3 py-1 rounded-full text-sm transition-colors ${
                        feedback.keywords_highlighted.includes(keyword)
                          ? 'bg-eu-blue-500 text-white'
                          : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      }`}
                    >
                      {keyword}
                    </button>
                  ))}
                </div>
              </div>

              {/* Prediction Review */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 mb-4">Prediction Review</h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Correct Queue
                    </label>
                    <select
                      value={feedback.corrected_queue}
                      onChange={(e) => setFeedback({...feedback, corrected_queue: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
                    >
                      <option value="CTI">CTI</option>
                      <option value="DFIR::incidents">DFIR::incidents</option>
                      <option value="DFIR::phishing">DFIR::phishing</option>
                      <option value="OFFSEC::CVD">OFFSEC::CVD</option>
                      <option value="OFFSEC::Pentesting">OFFSEC::Pentesting</option>
                      <option value="SMS">SMS</option>
                      <option value="Trash">Trash</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Difficulty Rating (1-5)
                    </label>
                    <div className="flex space-x-2">
                      {[1, 2, 3, 4, 5].map((rating) => (
                        <button
                          key={rating}
                          onClick={() => setFeedback({...feedback, difficulty_score: rating})}
                          className={`w-10 h-10 rounded-full border-2 flex items-center justify-center ${
                            feedback.difficulty_score === rating
                              ? 'border-eu-blue-500 bg-eu-blue-500 text-white'
                              : 'border-gray-300 text-gray-600 hover:border-gray-400'
                          }`}
                        >
                          {rating}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Feedback Notes
                    </label>
                    <textarea
                      value={feedback.feedback_notes}
                      onChange={(e) => setFeedback({...feedback, feedback_notes: e.target.value})}
                      placeholder="Add any notes about this classification..."
                      rows={3}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Was the prediction correct?
                    </label>
                    <div className="flex space-x-4">
                      <button
                        onClick={() => setFeedback({...feedback, is_correct: true})}
                        className={`flex items-center px-4 py-2 rounded-lg border-2 transition-colors ${
                          feedback.is_correct === true
                            ? 'border-green-500 bg-green-50 text-green-700'
                            : 'border-gray-300 text-gray-600 hover:border-green-300'
                        }`}
                      >
                        <ThumbsUp className="w-4 h-4 mr-2" />
                        Correct
                      </button>
                      <button
                        onClick={() => setFeedback({...feedback, is_correct: false})}
                        className={`flex items-center px-4 py-2 rounded-lg border-2 transition-colors ${
                          feedback.is_correct === false
                            ? 'border-red-500 bg-red-50 text-red-700'
                            : 'border-gray-300 text-gray-600 hover:border-red-300'
                        }`}
                      >
                        <ThumbsDown className="w-4 h-4 mr-2" />
                        Incorrect
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <div className="flex justify-end">
                <button
                  onClick={submitFeedback}
                  disabled={submitting || feedback.is_correct === null}
                  className="eu-button-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                >
                  {submitting ? (
                    <Loader className="w-4 h-4 animate-spin mr-2" />
                  ) : (
                    <Save className="w-4 h-4 mr-2" />
                  )}
                  Submit Feedback
                </button>
              </div>
            </div>
          ) : (
            <div className="eu-card p-12 text-center">
              <AlertTriangle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No ticket selected</h3>
              <p className="text-gray-600">Select a ticket from the list to start reviewing</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HumanReview;
