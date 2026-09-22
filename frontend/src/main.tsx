import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";

import "./index.css";


/*
 * Тема выставляется ещё до render().
 * Поэтому страница не успевает сначала
 * показать одну тему, а потом другую.
 */

const savedTheme =
  localStorage.getItem(
    "theme"
  );

const initialTheme =
  savedTheme === "light"
    ? "light"
    : "dark";

document.documentElement.dataset.theme =
  initialTheme;


const root =
  document.getElementById(
    "root"
  );


if (!root) {
  throw new Error(
    "Root element not found"
  );
}


createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>
);