/**
 * Test suite for TicketAnalyzer component
 * Tests single ticket analysis, batch processing, and file upload functionality
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import axios from 'axios';
import TicketAnalyzer from '../TicketAnalyzer';

// Mock axios
jest.mock('axios');
const mockedAxios = axios;

// Mock toast notifications
jest.mock('react-hot-toast', () => ({
  success: jest.fn(),
  error: jest.fn(),
}));

const renderWithRouter = (component) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  );
};

describe('TicketAnalyzer Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Single Ticket Analysis', () => {
    test('renders single ticket form', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      expect(screen.getByText('Single Ticket Analysis')).toBeInTheDocument();
      expect(screen.getByLabelText(/title/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/priority/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/category/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /analyze ticket/i })).toBeInTheDocument();
    });

    test('handles form input changes', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      const titleInput = screen.getByLabelText(/title/i);
      const descriptionInput = screen.getByLabelText(/description/i);
      
      fireEvent.change(titleInput, { target: { value: 'Test Ticket' } });
      fireEvent.change(descriptionInput, { target: { value: 'Test description' } });
      
      expect(titleInput.value).toBe('Test Ticket');
      expect(descriptionInput.value).toBe('Test description');
    });

    test('validates required fields', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      const analyzeButton = screen.getByRole('button', { name: /analyze ticket/i });
      fireEvent.click(analyzeButton);
      
      // Should show validation errors
      expect(screen.getByText(/title is required/i)).toBeInTheDocument();
      expect(screen.getByText(/description is required/i)).toBeInTheDocument();
    });

    test('submits single ticket analysis successfully', async () => {
      const mockResponse = {
        data: {
          prediction_id: 'test-123',
          predicted_queue: 'CTI',
          confidence_score: 0.85,
          routing_decision: 'auto_route',
          reasoning: 'High confidence prediction'
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      renderWithRouter(<TicketAnalyzer />);
      
      // Fill form
      fireEvent.change(screen.getByLabelText(/title/i), { target: { value: 'Test Ticket' } });
      fireEvent.change(screen.getByLabelText(/description/i), { target: { value: 'Test description' } });
      fireEvent.change(screen.getByLabelText(/priority/i), { target: { value: 'high' } });
      fireEvent.change(screen.getByLabelText(/category/i), { target: { value: 'security' } });
      
      // Submit form
      fireEvent.click(screen.getByRole('button', { name: /analyze ticket/i }));
      
      await waitFor(() => {
        expect(mockedAxios.post).toHaveBeenCalledWith('/api/predict', {
          title: 'Test Ticket',
          description: 'Test description',
          priority: 'high',
          category: 'security'
        });
      });
      
      // Should show results
      expect(screen.getByText('CTI')).toBeInTheDocument();
      expect(screen.getByText('85%')).toBeInTheDocument();
    });

    test('handles single ticket analysis error', async () => {
      mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));
      
      renderWithRouter(<TicketAnalyzer />);
      
      // Fill and submit form
      fireEvent.change(screen.getByLabelText(/title/i), { target: { value: 'Test Ticket' } });
      fireEvent.change(screen.getByLabelText(/description/i), { target: { value: 'Test description' } });
      fireEvent.click(screen.getByRole('button', { name: /analyze ticket/i }));
      
      await waitFor(() => {
        expect(screen.getByText(/failed to analyze ticket/i)).toBeInTheDocument();
      });
    });
  });

  describe('Batch Processing', () => {
    test('renders batch upload section', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      expect(screen.getByText('Batch Processing')).toBeInTheDocument();
      expect(screen.getByText(/drag and drop/i)).toBeInTheDocument();
      expect(screen.getByText(/json or jsonl/i)).toBeInTheDocument();
    });

    test('handles file selection', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      const fileInput = screen.getByLabelText(/upload file/i);
      const file = new File(['{"id": "test", "title": "Test"}'], 'test.jsonl', {
        type: 'application/json'
      });
      
      fireEvent.change(fileInput, { target: { files: [file] } });
      
      expect(screen.getByText('test.jsonl')).toBeInTheDocument();
    });

    test('validates file type', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      const fileInput = screen.getByLabelText(/upload file/i);
      const invalidFile = new File(['content'], 'test.txt', {
        type: 'text/plain'
      });
      
      fireEvent.change(fileInput, { target: { files: [invalidFile] } });
      
      expect(screen.getByText(/please upload a json or jsonl file/i)).toBeInTheDocument();
    });

    test('validates file size', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      const fileInput = screen.getByLabelText(/upload file/i);
      const largeFile = new File(['x'.repeat(10000000)], 'large.jsonl', {
        type: 'application/json'
      });
      
      // Mock file size
      Object.defineProperty(largeFile, 'size', { value: 10000000 });
      
      fireEvent.change(fileInput, { target: { files: [largeFile] } });
      
      expect(screen.getByText(/file size must be less than 50mb/i)).toBeInTheDocument();
    });

    test('processes batch file successfully', async () => {
      const mockResponse = {
        data: {
          message: 'Batch analysis completed successfully',
          total_processed: 2,
          successful_predictions: 2,
          failed_predictions: 0
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      renderWithRouter(<TicketAnalyzer />);
      
      // Upload file
      const fileInput = screen.getByLabelText(/upload file/i);
      const file = new File(['{"id": "test1", "title": "Test1"}\n{"id": "test2", "title": "Test2"}'], 'test.jsonl', {
        type: 'application/json'
      });
      
      fireEvent.change(fileInput, { target: { files: [file] } });
      
      // Process batch
      fireEvent.click(screen.getByRole('button', { name: /analyze batch/i }));
      
      await waitFor(() => {
        expect(mockedAxios.post).toHaveBeenCalledWith(
          '/api/predict/batch',
          expect.any(FormData),
          expect.objectContaining({
            headers: expect.objectContaining({
              'Content-Type': 'multipart/form-data'
            })
          })
        );
      });
      
      expect(screen.getByText(/batch analysis completed successfully/i)).toBeInTheDocument();
    });

    test('handles batch processing error', async () => {
      mockedAxios.post.mockRejectedValueOnce(new Error('Batch processing failed'));
      
      renderWithRouter(<TicketAnalyzer />);
      
      // Upload and process file
      const fileInput = screen.getByLabelText(/upload file/i);
      const file = new File(['{"id": "test", "title": "Test"}'], 'test.jsonl', {
        type: 'application/json'
      });
      
      fireEvent.change(fileInput, { target: { files: [file] } });
      fireEvent.click(screen.getByRole('button', { name: /analyze batch/i }));
      
      await waitFor(() => {
        expect(screen.getByText(/batch analysis failed/i)).toBeInTheDocument();
      });
    });

    test('handles drag and drop', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      const dropZone = screen.getByTestId('drop-zone');
      const file = new File(['{"id": "test", "title": "Test"}'], 'test.jsonl', {
        type: 'application/json'
      });
      
      fireEvent.dragOver(dropZone);
      fireEvent.drop(dropZone, { dataTransfer: { files: [file] } });
      
      expect(screen.getByText('test.jsonl')).toBeInTheDocument();
    });

    test('prevents default drag behaviors', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      const dropZone = screen.getByTestId('drop-zone');
      
      const dragOverEvent = new Event('dragover', { bubbles: true });
      const preventDefault = jest.fn();
      dragOverEvent.preventDefault = preventDefault;
      
      fireEvent(dropZone, dragOverEvent);
      
      expect(preventDefault).toHaveBeenCalled();
    });
  });

  describe('Results Display', () => {
    test('displays single ticket results', async () => {
      const mockResponse = {
        data: {
          prediction_id: 'test-123',
          predicted_queue: 'DFIR::incidents',
          confidence_score: 0.92,
          routing_decision: 'auto_route',
          reasoning: 'Security incident detected'
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      renderWithRouter(<TicketAnalyzer />);
      
      // Fill and submit form
      fireEvent.change(screen.getByLabelText(/title/i), { target: { value: 'Security Alert' } });
      fireEvent.change(screen.getByLabelText(/description/i), { target: { value: 'Malware detected' } });
      fireEvent.click(screen.getByRole('button', { name: /analyze ticket/i }));
      
      await waitFor(() => {
        expect(screen.getByText('DFIR::incidents')).toBeInTheDocument();
        expect(screen.getByText('92%')).toBeInTheDocument();
        expect(screen.getByText('auto_route')).toBeInTheDocument();
        expect(screen.getByText('Security incident detected')).toBeInTheDocument();
      });
    });

    test('displays batch results', async () => {
      const mockResponse = {
        data: {
          message: 'Batch analysis completed successfully',
          total_processed: 5,
          successful_predictions: 4,
          failed_predictions: 1
        }
      };
      
      mockedAxios.post.mockResolvedValueOnce(mockResponse);
      
      renderWithRouter(<TicketAnalyzer />);
      
      // Upload and process file
      const fileInput = screen.getByLabelText(/upload file/i);
      const file = new File(['{"id": "test", "title": "Test"}'], 'test.jsonl', {
        type: 'application/json'
      });
      
      fireEvent.change(fileInput, { target: { files: [file] } });
      fireEvent.click(screen.getByRole('button', { name: /analyze batch/i }));
      
      await waitFor(() => {
        expect(screen.getByText(/5 tickets processed/i)).toBeInTheDocument();
        expect(screen.getByText(/4 successful/i)).toBeInTheDocument();
        expect(screen.getByText(/1 failed/i)).toBeInTheDocument();
      });
    });

    test('clears results when new analysis starts', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      // Fill form to show results
      fireEvent.change(screen.getByLabelText(/title/i), { target: { value: 'Test' } });
      fireEvent.change(screen.getByLabelText(/description/i), { target: { value: 'Test description' } });
      
      // Clear form
      fireEvent.change(screen.getByLabelText(/title/i), { target: { value: '' } });
      fireEvent.change(screen.getByLabelText(/description/i), { target: { value: '' } });
      
      // Results should be cleared
      expect(screen.queryByText(/prediction/i)).not.toBeInTheDocument();
    });
  });

  describe('Loading States', () => {
    test('shows loading state during single analysis', async () => {
      mockedAxios.post.mockImplementation(() => new Promise(resolve => setTimeout(resolve, 100)));
      
      renderWithRouter(<TicketAnalyzer />);
      
      fireEvent.change(screen.getByLabelText(/title/i), { target: { value: 'Test' } });
      fireEvent.change(screen.getByLabelText(/description/i), { target: { value: 'Test description' } });
      fireEvent.click(screen.getByRole('button', { name: /analyze ticket/i }));
      
      expect(screen.getByText(/analyzing/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /analyze ticket/i })).toBeDisabled();
    });

    test('shows loading state during batch processing', async () => {
      mockedAxios.post.mockImplementation(() => new Promise(resolve => setTimeout(resolve, 100)));
      
      renderWithRouter(<TicketAnalyzer />);
      
      const fileInput = screen.getByLabelText(/upload file/i);
      const file = new File(['{"id": "test", "title": "Test"}'], 'test.jsonl', {
        type: 'application/json'
      });
      
      fireEvent.change(fileInput, { target: { files: [file] } });
      fireEvent.click(screen.getByRole('button', { name: /analyze batch/i }));
      
      expect(screen.getByText(/processing batch/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /analyze batch/i })).toBeDisabled();
    });
  });

  describe('Accessibility', () => {
    test('has proper form labels', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      expect(screen.getByLabelText(/title/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/priority/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/category/i)).toBeInTheDocument();
    });

    test('has proper button labels', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      expect(screen.getByRole('button', { name: /analyze ticket/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /analyze batch/i })).toBeInTheDocument();
    });

    test('supports keyboard navigation', () => {
      renderWithRouter(<TicketAnalyzer />);
      
      const titleInput = screen.getByLabelText(/title/i);
      const descriptionInput = screen.getByLabelText(/description/i);
      
      titleInput.focus();
      expect(titleInput).toHaveFocus();
      
      fireEvent.keyDown(titleInput, { key: 'Tab' });
      expect(descriptionInput).toHaveFocus();
    });
  });
});
