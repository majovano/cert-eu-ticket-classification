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
    <div className="min-h-screen bg-gray-50">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 flex flex-col ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        {/* Header */}
        <div className="flex items-center justify-between h-16 px-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 eu-gradient rounded-lg flex items-center justify-center">
              <Star className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-eu-blue-700">CERT-EU</h1>
              <p className="text-xs text-gray-500">Ticket Classification</p>
            </div>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="mt-8 px-4">
          <ul className="space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <li key={item.name}>
                  <Link
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

        {/* Footer */}
        <div className="mt-auto p-6 border-t border-gray-200">
          <div className="text-xs text-gray-500 text-center">
            <p>European Union</p>
            <p>Cybersecurity Emergency Response Team</p>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <div className="bg-white shadow-sm border-b border-gray-200">
          <div className="flex items-center justify-between h-16 px-6">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-600"
            >
              <Menu className="w-6 h-6" />
            </button>
            
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                <span className="font-medium">Welcome back,</span>
                <span className="ml-1 text-eu-blue-600">CERT Analyst</span>
              </div>
              <div className="w-8 h-8 bg-eu-blue-100 rounded-full flex items-center justify-center">
                <span className="text-sm font-medium text-eu-blue-700">CA</span>
              </div>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="p-6">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
