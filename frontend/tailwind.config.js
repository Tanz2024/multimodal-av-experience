/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Poppins"', 'sans-serif'],
      },
      colors: {
        gold: '#c7a76a',
        gold2: '#e5cf97',
        ink: '#060607',
      },
      boxShadow: {
        luxe: '0 40px 90px rgba(0,0,0,0.75)',
      },
    },
  },
  plugins: [],
};
