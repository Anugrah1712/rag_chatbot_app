// Chatbot.js
import React, { useState, useEffect, useRef } from "react";
import "./Chatbot.css";
import "./DeveloperConsole.css";
import CloseIcon from "@mui/icons-material/Close";
import SettingsIcon from "@mui/icons-material/Settings";
import DeveloperConsole from "./DeveloperConsole";
import SendIcon from "@mui/icons-material/Send";

function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [adminKey, setAdminKey] = useState("");
  const [accessGranted, setAccessGranted] = useState(false);
  const [showDeveloperConsole, setShowDeveloperConsole] = useState(false);
  const [messages, setMessages] = useState([
    { text: "Hello! 👋", sender: "bot" },
    { text: "How can I help you today?", sender: "bot" },
  ]);
  const [inputText, setInputText] = useState("");
  const chatBodyRef = useRef(null);

  // Auto-scroll chat body when messages update
  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [messages]);

  // Toggle chatbot visibility
  const toggleChatbot = () => {
    setIsOpen((prev) => !prev);
    setShowSettings(false);
    setShowDeveloperConsole(false);
  };

  // Open chatbot directly
  const openChatbot = () => {
    setIsOpen(true);
  };

  // Handle settings button click
  const handleSettingsClick = () => {
    setShowSettings(true);
  };

  // Handle admin key verification
  const handleAdminSubmit = () => {
    if (adminKey === "admin123") {
      setAccessGranted(true);
      setShowSettings(false);
      setShowDeveloperConsole(true);
    } else {
      alert("Incorrect Admin Key!");
    }
  };

  // Send user message to chatbot
  const handleSendMessage = async () => {
    if (inputText.trim() === "") return;

    const newMessages = [...messages, { text: inputText, sender: "user" }];
    setMessages(newMessages);
    setInputText("");

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({ prompt: inputText }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessages([...newMessages, { text: data.response, sender: "bot" }]);
      } else {
        setMessages([...newMessages, { text: "Error: " + data.detail, sender: "bot" }]);
      }
    } catch (error) {
      console.error("Chatbot error:", error);
      setMessages([...newMessages, { text: "Error connecting to server.", sender: "bot" }]);
    }
  };

  return (
    <div className="chatbot-container">
      <button className="chat-icon" onClick={toggleChatbot}>
        💬
      </button>

      {isOpen && (
        <div className="chatbox">
          <div className="chat-header">
            <span>AI Chat Assistant</span>
            <div className="icons">
              <SettingsIcon className="icon" onClick={handleSettingsClick} />
              <CloseIcon className="icon" onClick={toggleChatbot} />
            </div>
          </div>

          <div className="chat-body" ref={chatBodyRef}>
            {showDeveloperConsole ? (
              <DeveloperConsole
                closeConsole={() => {
                  setAccessGranted(false);
                  setShowDeveloperConsole(false);
                }}
                openChatbot={openChatbot} // ✅ Ensure chatbot opens correctly
              />
            ) : (
              messages.map((msg, index) => (
                <div
                  key={index}
                  className={msg.sender === "bot" ? "bot-message" : "user-message"}
                >
                  {msg.text}
                </div>
              ))
            )}
          </div>

          {!showDeveloperConsole && (
            <div className="chat-input-container">
              <input
                type="text"
                className="chat-input"
                placeholder="Type a message..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
              />
              <button className="send-button" onClick={handleSendMessage}>
                <SendIcon />
              </button>
            </div>
          )}

          {showSettings && (
            <div className="admin-modal">
              <div className="admin-modal-content">
                <div className="admin-header">
                  <h3>Enter Admin Key</h3>
                  <CloseIcon className="close-icon" onClick={() => setShowSettings(false)} />
                </div>
                <input
                  type="password"
                  value={adminKey}
                  onChange={(e) => setAdminKey(e.target.value)}
                  placeholder="Enter admin key"
                  className="admin-input"
                />
                <button className="admin-submit" onClick={handleAdminSubmit}>
                  Submit
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Chatbot;
