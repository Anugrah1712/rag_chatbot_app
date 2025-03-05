import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Chatbot from "./components/Chatbot";
import DeveloperConsole from "./components/DeveloperConsole"; // Import DeveloperConsole
import "./App.css";

function App() {
  return (
    <Router>
      <div className="app">
        <h1 className="welcome-text">Welcome to my Website</h1>
        <Chatbot />

        {/* Define Routes */}
        <Routes>
          <Route path="/developer-console" element={<DeveloperConsole />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
