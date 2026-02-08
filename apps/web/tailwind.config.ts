import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: "#1a1a1b",
        border: "#2d2d2e",
        muted: "#71767b",
      },
    },
  },
  plugins: [],
};

export default config;
