/** @type {import('tailwindcss').Config} */

const withMT = require("@material-tailwind/react/utils/withMT");  // added for using Material Tailwind

module.exports = withMT({
  content: [
    "./src/**/*.{html,js,jsx,ts,tsx}",
    "./node_modules/@material-tailwind/react/components/**/*.{js,ts,jsx,tsx}",
    "./node_modules/@material-tailwind/react/theme/components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        body: "Poppins, sans-serif",
      },
      colors: {
        'cool-gray': {
          900: '#f3f3f3',
        },
      },
    },
  },
  plugins: [],
});

