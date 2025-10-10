import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import TicketAnalyzer from './pages/TicketAnalyzer';
import QueueAnalysis from './pages/QueueAnalysis';
import HumanReview from './pages/HumanReview';
import Reports from './pages/Reports';
import TimeSeriesForecast from './pages/TimeSeriesForecast';
import './index.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analyzer" element={<TicketAnalyzer />} />
            <Route path="/queues" element={<QueueAnalysis />} />
            <Route path="/review" element={<HumanReview />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/forecast" element={<TimeSeriesForecast />} />
          </Routes>
        </Layout>
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              style: {
                background: '#2e7d32',
              },
            },
            error: {
              style: {
                background: '#c62828',
              },
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;
