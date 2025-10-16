/**
 * Test suite for Layout component
 * Tests navigation, routing, and component rendering
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Layout from '../Layout';

// Mock the pages
jest.mock('../../pages/Dashboard', () => {
  return function MockDashboard() {
    return <div data-testid="dashboard-page">Dashboard Page</div>;
  };
});

jest.mock('../../pages/TicketAnalyzer', () => {
  return function MockTicketAnalyzer() {
    return <div data-testid="ticket-analyzer-page">Ticket Analyzer Page</div>;
  };
});

jest.mock('../../pages/TimeSeriesForecast', () => {
  return function MockTimeSeriesForecast() {
    return <div data-testid="time-series-page">Time Series Forecast Page</div>;
  };
});

jest.mock('../../pages/HumanReview', () => {
  return function MockHumanReview() {
    return <div data-testid="human-review-page">Human Review Page</div>;
  };
});

jest.mock('../../pages/QueueAnalysis', () => {
  return function MockQueueAnalysis() {
    return <div data-testid="queue-analysis-page">Queue Analysis Page</div>;
  };
});

jest.mock('../../pages/Reports', () => {
  return function MockReports() {
    return <div data-testid="reports-page">Reports Page</div>;
  };
});

const renderWithRouter = (component) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  );
};

describe('Layout Component', () => {
  test('renders navigation menu', () => {
    renderWithRouter(<Layout />);
    
    expect(screen.getByText('CERT-EU Ticket Classification')).toBeInTheDocument();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Ticket Analyzer')).toBeInTheDocument();
    expect(screen.getByText('Time Series Forecast')).toBeInTheDocument();
    expect(screen.getByText('Human Review')).toBeInTheDocument();
    expect(screen.getByText('Queue Analysis')).toBeInTheDocument();
    expect(screen.getByText('Reports')).toBeInTheDocument();
  });

  test('renders dashboard by default', () => {
    renderWithRouter(<Layout />);
    expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
  });

  test('navigates to different pages when menu items are clicked', () => {
    renderWithRouter(<Layout />);
    
    // Click on Ticket Analyzer
    fireEvent.click(screen.getByText('Ticket Analyzer'));
    expect(screen.getByTestId('ticket-analyzer-page')).toBeInTheDocument();
    
    // Click on Time Series Forecast
    fireEvent.click(screen.getByText('Time Series Forecast'));
    expect(screen.getByTestId('time-series-page')).toBeInTheDocument();
    
    // Click on Human Review
    fireEvent.click(screen.getByText('Human Review'));
    expect(screen.getByTestId('human-review-page')).toBeInTheDocument();
    
    // Click on Queue Analysis
    fireEvent.click(screen.getByText('Queue Analysis'));
    expect(screen.getByTestId('queue-analysis-page')).toBeInTheDocument();
    
    // Click on Reports
    fireEvent.click(screen.getByText('Reports'));
    expect(screen.getByTestId('reports-page')).toBeInTheDocument();
    
    // Click on Dashboard
    fireEvent.click(screen.getByText('Dashboard'));
    expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
  });

  test('shows active navigation state', () => {
    renderWithRouter(<Layout />);
    
    // Dashboard should be active by default
    const dashboardLink = screen.getByText('Dashboard').closest('a');
    expect(dashboardLink).toHaveClass('bg-blue-100');
    
    // Click on another page
    fireEvent.click(screen.getByText('Ticket Analyzer'));
    const ticketAnalyzerLink = screen.getByText('Ticket Analyzer').closest('a');
    expect(ticketAnalyzerLink).toHaveClass('bg-blue-100');
  });

  test('renders mobile menu toggle', () => {
    renderWithRouter(<Layout />);
    
    // Check if mobile menu button exists
    const mobileMenuButton = screen.getByRole('button', { name: /menu/i });
    expect(mobileMenuButton).toBeInTheDocument();
  });

  test('toggles mobile menu when button is clicked', () => {
    renderWithRouter(<Layout />);
    
    const mobileMenuButton = screen.getByRole('button', { name: /menu/i });
    
    // Menu should be hidden by default
    const mobileMenu = screen.getByTestId('mobile-menu');
    expect(mobileMenu).toHaveClass('hidden');
    
    // Click to open menu
    fireEvent.click(mobileMenuButton);
    expect(mobileMenu).not.toHaveClass('hidden');
    
    // Click to close menu
    fireEvent.click(mobileMenuButton);
    expect(mobileMenu).toHaveClass('hidden');
  });

  test('closes mobile menu when navigation item is clicked', () => {
    renderWithRouter(<Layout />);
    
    const mobileMenuButton = screen.getByRole('button', { name: /menu/i });
    const mobileMenu = screen.getByTestId('mobile-menu');
    
    // Open mobile menu
    fireEvent.click(mobileMenuButton);
    expect(mobileMenu).not.toHaveClass('hidden');
    
    // Click on a navigation item
    fireEvent.click(screen.getByText('Ticket Analyzer'));
    
    // Menu should be closed
    expect(mobileMenu).toHaveClass('hidden');
    expect(screen.getByTestId('ticket-analyzer-page')).toBeInTheDocument();
  });

  test('renders with correct CSS classes', () => {
    renderWithRouter(<Layout />);
    
    const layout = screen.getByTestId('layout');
    expect(layout).toHaveClass('min-h-screen', 'bg-gray-50');
    
    const sidebar = screen.getByTestId('sidebar');
    expect(sidebar).toHaveClass('bg-white', 'shadow-lg');
    
    const mainContent = screen.getByTestId('main-content');
    expect(mainContent).toHaveClass('flex-1', 'p-6');
  });

  test('handles window resize for mobile responsiveness', () => {
    renderWithRouter(<Layout />);
    
    // Mock window.innerWidth
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 768, // Mobile width
    });
    
    // Trigger resize event
    fireEvent(window, new Event('resize'));
    
    // Component should handle resize gracefully
    expect(screen.getByTestId('layout')).toBeInTheDocument();
  });
});
