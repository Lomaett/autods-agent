/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,jsx}",
    "./components/**/*.{js,jsx}",
    "./lib/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        ocean: "#0e7490",
        ember: "#b45309",
        fog: "#f1f5f9",
        mint: "#0f766e"
      },
      boxShadow: {
        panel: "0 10px 35px -15px rgba(15, 23, 42, 0.35)"
      }
    }
  },
  plugins: [require("@tailwindcss/typography")]
};
