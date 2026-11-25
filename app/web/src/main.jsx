import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "@fontsource-variable/unbounded";
import "@fontsource-variable/manrope";
import "./style.css";

const root = document.getElementById("root");
if (!root) {
  throw new Error('Missing root element in index.html');
}

createRoot(root).render(<App />);
