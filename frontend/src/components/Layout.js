import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  BarChart3, 
  FileText, 
  Shield, 
  Users, 
  FileBarChart,
  TrendingUp,
  Menu,
  X,
  Star
} from 'lucide-react';

const Layout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: BarChart3 },
    { name: 'Ticket Analyzer', href: '/analyzer', icon: FileText },
    { name: 'Queue Analysis', href: '/queues', icon: Shield },
    { name: 'Human Review', href: '/review', icon: Users },
    { name: 'Reports', href: '/reports', icon: FileBarChart },
    { name: 'Time Series Forecast', href: '/forecast', icon: TrendingUp },
  ];

  const isActive = (href) => {
    // Handle root path specially
    if (href === '/' && location.pathname === '/') return true;
    if (href === '/' && location.pathname !== '/') return false;
    // For other paths, check if they match
    return location.pathname === href;
  };

  return (
    <div id="main-container" className="min-h-screen bg-gray-50 flex">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          id="mobile-sidebar-overlay"
          className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div id="sidebar" className={`fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 flex flex-col ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        {/* Header */}
        <div id="sidebar-header" className="h-16 px-6 border-b border-gray-200">
          <div id="sidebar-header-content" className="flex items-center justify-between h-full">
            <div id="sidebar-logo-section" className="flex items-center space-x-3">
              <div id="sidebar-logo-icon" className="w-8 h-8 eu-gradient rounded-lg flex items-center justify-center">
                <Star className="w-5 h-5 text-white" />
              </div>
              <div id="sidebar-logo-text">
                <h1 className="text-lg font-bold text-eu-blue-700">CERT-EU</h1>
                <p className="text-xs text-gray-500">Ticket Classification</p>
              </div>
            </div>
            <button
              id="sidebar-close-button"
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-600"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Navigation */}
        <nav id="sidebar-navigation" className="mt-8 px-4">
          <ul id="sidebar-navigation-list" className="space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <li key={item.name} id={`nav-item-${item.name.toLowerCase().replace(/\s+/g, '-')}`}>
                  <Link
                    id={`nav-link-${item.name.toLowerCase().replace(/\s+/g, '-')}`}
                    to={item.href}
                    className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors duration-200 ${
                      isActive(item.href)
                        ? 'bg-eu-blue-50 text-eu-blue-700 border-r-2 border-eu-blue-500'
                        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                    }`}
                    onClick={() => {
                      console.log('Navigating to:', item.href);
                      setSidebarOpen(false);
                    }}
                  >
                    <Icon className={`w-5 h-5 mr-3 ${
                      isActive(item.href) ? 'text-eu-blue-600' : 'text-gray-400'
                    }`} />
                    {item.name}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* User Greeting - positioned below navigation */}
        <div id="sidebar-user-greeting" className="px-4 py-4 border-t border-gray-200">
          <div id="sidebar-user-greeting-content" className="flex items-center space-x-3">
            <div id="sidebar-user-avatar" className="w-8 h-8 bg-eu-blue-100 rounded-full flex items-center justify-center">
              <span className="text-sm font-medium text-eu-blue-700">CA</span>
            </div>
            <div id="sidebar-user-text" className="text-sm text-gray-600">
              <span className="font-medium">Welcome back,</span>
              <span className="ml-1 text-eu-blue-600">CERT Analyst</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div id="sidebar-footer" className="mt-auto border-t border-gray-200">
          {/* Footer Text */}
          <div id="sidebar-footer-content" className="p-4 text-xs text-gray-500 text-center">
            <p id="sidebar-footer-line1">European Union</p>
            <p id="sidebar-footer-line2">Cybersecurity Emergency Response Team</p>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div id="main-content" className="flex-1 flex flex-col">
        {/* Page content */}
        <main id="main-page-content" className="flex-1 p-6">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
