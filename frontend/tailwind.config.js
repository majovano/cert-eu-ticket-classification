/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        // EU Official Colors
        'eu-blue': {
          50: '#e6f3ff',
          100: '#b3d9ff',
          200: '#80bfff',
          300: '#4da6ff',
          400: '#1a8cff',
          500: '#003399', // Primary EU Blue
          600: '#002980',
          700: '#001f66',
          800: '#00144d',
          900: '#000a33'
        },
        'eu-yellow': {
          50: '#fffbf0',
          100: '#fff3d6',
          200: '#ffebbc',
          300: '#ffe3a2',
          400: '#ffdb88',
          500: '#FFCC02', // Primary EU Yellow
          600: '#e6b800',
          700: '#cc9f00',
          800: '#b38600',
          900: '#996d00'
        },
        'eu-gray': {
          50: '#f8f9fa',
          100: '#e9ecef',
          200: '#dee2e6',
          300: '#ced4da',
          400: '#adb5bd',
          500: '#6c757d',
          600: '#495057',
          700: '#343a40',
          800: '#212529',
          900: '#1a1e21'
        }
      },
      fontFamily: {
        'eu': ['Inter', 'system-ui', 'sans-serif']
      },
      boxShadow: {
        'eu': '0 4px 6px -1px rgba(0, 51, 153, 0.1), 0 2px 4px -1px rgba(0, 51, 153, 0.06)'
      }
    }
  },
  plugins: []
}
