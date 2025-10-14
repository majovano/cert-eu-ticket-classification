import React, { useState, useEffect } from 'react';
import { 
  Download, 
  FileText, 
  CheckCircle,
  AlertTriangle,
  Clock,
  FileImage,
  Mail,
  Calendar,
  Filter
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import html2canvas from 'html2canvas';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const Reports = () => {
  const [dashboardStats, setDashboardStats] = useState(null);
  const [queuePerformance, setQueuePerformance] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [emailData, setEmailData] = useState({
    recipient_email: '',
    subject: 'CERT-EU Ticket Classification Report',
    message: '',
    date_from: '',
    date_to: '',
    ticket_limit: ''
  });
  const [filteredData, setFilteredData] = useState(null);

  useEffect(() => {
    fetchReportData();
  }, []);

  const fetchReportData = async () => {
    try {
      setLoading(true);
      console.log('Fetching report data...');
      
      const [statsResponse, queueResponse] = await Promise.all([
        axios.get('/api/dashboard/stats'),
        axios.get('/api/dashboard/queue-performance')
      ]);
      
      console.log('Data received:', statsResponse.data, queueResponse.data);
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
    if (!dashboardStats || !queuePerformance.length) {
      toast.error('No data available to generate report');
      return;
    }

    const reportData = {
      generated_at: new Date().toISOString(),
      stats: dashboardStats,
      queue_performance: queuePerformance
    };

    const dataStr = JSON.stringify(reportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `cert-eu-report-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    toast.success('Report downloaded successfully');
  };

  const generatePDFReport = async () => {
    if (!dashboardStats || !queuePerformance.length) {
      toast.error('No data available to generate PDF report');
      return;
    }

    try {
      console.log('Starting PDF generation...');
      console.log('Dashboard stats:', dashboardStats);
      console.log('Queue performance:', queuePerformance);
      
      toast.loading('Generating PDF report...', { id: 'pdf-generation' });
      
      const pdf = new jsPDF('p', 'mm', 'a4');
      
      // EU Colors
      const euBlue = [3, 51, 153]; // RGB values for jsPDF
      
      // Title Page
      pdf.setFillColor(euBlue[0], euBlue[1], euBlue[2]);
      pdf.rect(0, 0, 210, 40, 'F');
      
      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(24);
      pdf.setFont('helvetica', 'bold');
      pdf.text('CERT-EU', 20, 20);
      pdf.text('Ticket Classification Report', 20, 30);
      
      pdf.setTextColor(0, 0, 0);
      pdf.setFontSize(16);
      pdf.setFont('helvetica', 'normal');
      pdf.text('European Union Cybersecurity Emergency Response Team', 20, 50);
      
      pdf.setFontSize(12);
      pdf.text(`Generated on: ${new Date().toLocaleDateString()}`, 20, 65);
      pdf.text(`Report Period: Last 30 days`, 20, 75);
      
      // Executive Summary
      pdf.setFontSize(18);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Executive Summary', 20, 95);
      
      pdf.setFontSize(11);
      pdf.setFont('helvetica', 'normal');
      
      // Safely calculate percentages
      const autoRoutePercent = dashboardStats.total_tickets > 0 ? 
        ((dashboardStats.auto_routed / dashboardStats.total_tickets) * 100).toFixed(1) : '0.0';
      const humanReviewPercent = dashboardStats.total_tickets > 0 ? 
        ((dashboardStats.human_verify / dashboardStats.total_tickets) * 100).toFixed(1) : '0.0';
      const manualTriagePercent = dashboardStats.total_tickets > 0 ? 
        ((dashboardStats.manual_triage / dashboardStats.total_tickets) * 100).toFixed(1) : '0.0';
      
      const summaryText = [
        `Total Tickets Processed: ${dashboardStats.total_tickets || 0}`,
        `Auto-Routed (High Confidence): ${dashboardStats.auto_routed || 0} (${autoRoutePercent}%)`,
        `Human Review Required: ${dashboardStats.human_verify || 0} (${humanReviewPercent}%)`,
        `Manual Triage Required: ${dashboardStats.manual_triage || 0} (${manualTriagePercent}%)`,
        `Average Confidence Score: ${((dashboardStats.avg_confidence || 0) * 100).toFixed(1)}%`
      ];
      
      summaryText.forEach((text, index) => {
        pdf.text(text, 20, 105 + (index * 8));
      });
      
      // Queue Performance Table
      pdf.setFontSize(18);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Queue Performance Analysis', 20, 160);
      
      // Safely map queue data
      const tableData = queuePerformance.map(queue => [
        queue.predicted_queue || 'Unknown',
        (queue.total_predictions || 0).toString(),
        `${((queue.avg_confidence || 0) * 100).toFixed(1)}%`,
        (queue.auto_routed_count || 0).toString(),
        (queue.human_verify_count || 0).toString(),
        (queue.manual_triage_count || 0).toString()
      ]);
      
      autoTable(pdf, {
        head: [['Queue', 'Total', 'Avg Confidence', 'Auto-Routed', 'Human Review', 'Manual Triage']],
        body: tableData,
        startY: 170,
        styles: { fontSize: 9 },
        headStyles: { fillColor: euBlue, textColor: 255 },
        alternateRowStyles: { fillColor: [245, 245, 245] }
      });
      
      // Key Insights
      const finalY = pdf.lastAutoTable ? pdf.lastAutoTable.finalY : 200;
      pdf.setFontSize(18);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Key Insights', 20, finalY + 20);
      
      pdf.setFontSize(11);
      pdf.setFont('helvetica', 'normal');
      const insights = [
        '• The system successfully processed all incoming tickets with high accuracy',
        '• Majority of tickets require human review, indicating good quality control',
        '• Auto-routing is functioning effectively for high-confidence predictions',
        '• Queue distribution shows balanced workload across different categories',
        '• Confidence scores indicate reliable ML model performance'
      ];
      
      insights.forEach((insight, index) => {
        pdf.text(insight, 20, finalY + 35 + (index * 8));
      });
      
      // Footer
      pdf.setFontSize(8);
      pdf.setTextColor(128, 128, 128);
      pdf.text('CERT-EU Ticket Classification System - Confidential Report', 105, 290, { align: 'center' });
      
      // Save the PDF
      pdf.save(`cert-eu-comprehensive-report-${new Date().toISOString().split('T')[0]}.pdf`);
      
      toast.success('PDF report generated successfully!', { id: 'pdf-generation' });
      
    } catch (error) {
      console.error('Error generating PDF:', error);
      console.error('Error details:', error.message);
      toast.error(`Failed to generate PDF report: ${error.message}`, { id: 'pdf-generation' });
    }
  };

  const sendEmailReport = async () => {
    if (!emailData.recipient_email) {
      toast.error('Please enter recipient email address');
      return;
    }

    try {
      toast.loading('Sending email report...', { id: 'email-sending' });
      
      const response = await axios.post('/api/reports/email', emailData);
      
      toast.success(`Email report sent successfully! ${response.data.message}`, { id: 'email-sending' });
      setShowEmailModal(false);
      
      // Reset email data
      setEmailData({
        recipient_email: '',
        subject: 'CERT-EU Ticket Classification Report',
        message: '',
        date_from: '',
        date_to: '',
        ticket_limit: ''
      });
      
    } catch (error) {
      console.error('Error sending email report:', error);
      toast.error(`Failed to send email report: ${error.response?.data?.detail || error.message}`, { id: 'email-sending' });
    }
  };

  const previewFilteredData = async () => {
    try {
      const params = new URLSearchParams();
      if (emailData.date_from) params.append('date_from', emailData.date_from);
      if (emailData.date_to) params.append('date_to', emailData.date_to);
      if (emailData.ticket_limit) params.append('ticket_limit', emailData.ticket_limit);
      
      const response = await axios.get(`/api/reports/filtered?${params.toString()}`);
      setFilteredData(response.data);
      
    } catch (error) {
      console.error('Error previewing filtered data:', error);
      toast.error('Failed to preview filtered data');
    }
  };

  const generatePeriodInfo = () => {
    if (emailData.date_from && emailData.date_to) {
      return `Period: ${emailData.date_from} to ${emailData.date_to}`;
    } else if (emailData.ticket_limit) {
      return `Latest ${emailData.ticket_limit} tickets`;
    } else {
      return 'All available tickets';
    }
  };

  // Chart data
  const queueChartData = {
    labels: queuePerformance.map(q => q.predicted_queue),
    datasets: [
      {
        label: 'Total Predictions',
        data: queuePerformance.map(q => q.total_predictions),
        backgroundColor: '#003399',
        borderColor: '#003399',
        borderWidth: 1,
      },
    ],
  };

  const routingChartData = {
    labels: ['Auto-Routed', 'Human Review', 'Manual Triage'],
    datasets: [
      {
        data: [dashboardStats?.auto_routed || 0, dashboardStats?.human_verify || 0, dashboardStats?.manual_triage || 0],
        backgroundColor: ['#10B981', '#F59E0B', '#EF4444'],
        borderColor: ['#10B981', '#F59E0B', '#EF4444'],
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Queue Distribution'
      },
    },
  };

  const routingChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Routing Decision Distribution'
      },
    },
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-eu-blue-500"></div>
        <span className="ml-3 text-gray-600">Loading report data...</span>
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
        <div className="flex space-x-3">
          <button
            onClick={generateReport}
            className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center transition-colors"
            disabled={!dashboardStats}
          >
            <FileText className="w-4 h-4 mr-2" />
            Export JSON
          </button>
          <button
            onClick={generatePDFReport}
            className="bg-eu-blue-500 hover:bg-eu-blue-600 text-white px-4 py-2 rounded-lg flex items-center transition-colors"
            disabled={!dashboardStats}
          >
            <FileImage className="w-4 h-4 mr-2" />
            Generate PDF Report
          </button>
          <button
            onClick={() => setShowEmailModal(true)}
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center transition-colors"
            disabled={!dashboardStats}
          >
            <Mail className="w-4 h-4 mr-2" />
            Send Email Report
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      {dashboardStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Tickets</p>
                <p className="text-2xl font-bold text-gray-900">{dashboardStats.total_tickets}</p>
                <p className="text-sm text-gray-500">All processed tickets</p>
              </div>
              <FileText className="w-8 h-8 text-eu-blue-500" />
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Auto-Routed</p>
                <p className="text-2xl font-bold text-green-600">{dashboardStats.auto_routed}</p>
                <p className="text-sm text-green-600 flex items-center">
                  <CheckCircle className="w-4 h-4 mr-1" />
                  High confidence
                </p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Human Review</p>
                <p className="text-2xl font-bold text-yellow-600">{dashboardStats.human_verify}</p>
                <p className="text-sm text-yellow-600 flex items-center">
                  <Clock className="w-4 h-4 mr-1" />
                  Medium confidence
                </p>
              </div>
              <Clock className="w-8 h-8 text-yellow-500" />
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Manual Triage</p>
                <p className="text-2xl font-bold text-red-600">{dashboardStats.manual_triage}</p>
                <p className="text-sm text-red-600 flex items-center">
                  <AlertTriangle className="w-4 h-4 mr-1" />
                  Low confidence
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </div>
        </div>
      )}

      {/* Queue Performance Table */}
      {queuePerformance.length > 0 && (
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Queue Performance Analysis</h3>
            <p className="text-sm text-gray-500 mt-1">Detailed breakdown by queue category</p>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Queue
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Total Predictions
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Avg Confidence
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Auto-Routed
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Human Review
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Manual Triage
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {queuePerformance.map((queue, index) => (
                  <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {queue.predicted_queue}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {queue.total_predictions}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                          <div 
                            className="bg-eu-blue-500 h-2 rounded-full" 
                            style={{ width: `${queue.avg_confidence * 100}%` }}
                          ></div>
                        </div>
                        <span>{(queue.avg_confidence * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600">
                      {queue.auto_routed_count}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-yellow-600">
                      {queue.human_verify_count}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600">
                      {queue.manual_triage_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Summary Stats */}
      {dashboardStats && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Report Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <p className="text-sm text-gray-500">Average Confidence</p>
              <p className="text-2xl font-bold text-eu-blue-600">
                {(dashboardStats.avg_confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-500">Auto-Route Rate</p>
              <p className="text-2xl font-bold text-green-600">
                {((dashboardStats.auto_routed / dashboardStats.total_tickets) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-500">Human Review Rate</p>
              <p className="text-2xl font-bold text-yellow-600">
                {((dashboardStats.human_verify / dashboardStats.total_tickets) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Charts Section */}
      {queuePerformance.length > 0 && dashboardStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Queue Distribution Chart */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Queue Distribution</h3>
            <div className="h-64">
              <Bar data={queueChartData} options={chartOptions} />
            </div>
          </div>
          
          {/* Routing Decision Chart */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Routing Decision Distribution</h3>
            <div className="h-64">
              <Doughnut data={routingChartData} options={routingChartOptions} />
            </div>
          </div>
        </div>
      )}

      {/* Email Modal */}
      {showEmailModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-900">Send Email Report</h2>
                <button
                  onClick={() => setShowEmailModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="space-y-4">
                {/* Email Address */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Recipient Email Address *
                  </label>
                  <input
                    type="email"
                    value={emailData.recipient_email}
                    onChange={(e) => setEmailData({...emailData, recipient_email: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
                    placeholder="recipient@example.com"
                  />
                </div>

                {/* Subject */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Subject
                  </label>
                  <input
                    type="text"
                    value={emailData.subject}
                    onChange={(e) => setEmailData({...emailData, subject: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
                  />
                </div>

                {/* Report Info */}
                <div className="border-t pt-4">
                  <h3 className="text-lg font-medium text-gray-900 mb-3 flex items-center">
                    <Calendar className="w-5 h-5 mr-2" />
                    Report Information
                  </h3>
                  
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-700">
                      <strong>📅 Report Date:</strong> {new Date().toLocaleDateString('en-GB', { 
                        day: 'numeric', 
                        month: 'long', 
                        year: 'numeric' 
                      })} (Today)
                    </p>
                    <p className="text-sm text-gray-700 mt-2">
                      <strong>📊 Data Scope:</strong> All available tickets in the system ({dashboardStats?.total_tickets || 0} tickets)
                    </p>
                    <p className="text-sm text-gray-600 mt-2">
                      This email will include a professional report with today's date, key metrics, and queue performance analysis.
                    </p>
                  </div>
                </div>

                {/* Custom Message */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Custom Message (optional)
                  </label>
                  <textarea
                    value={emailData.message}
                    onChange={(e) => setEmailData({...emailData, message: e.target.value})}
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-eu-blue-500 focus:border-transparent"
                    placeholder="Add any additional context or notes for the recipient..."
                  />
                </div>

                {/* Action Buttons */}
                <div className="flex space-x-3 pt-4 border-t">
                  <button
                    onClick={() => setShowEmailModal(false)}
                    className="flex-1 bg-gray-300 hover:bg-gray-400 text-gray-700 px-4 py-2 rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={sendEmailReport}
                    className="flex-1 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center justify-center transition-colors"
                  >
                    <Mail className="w-4 h-4 mr-2" />
                    Send Email Report
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Reports;