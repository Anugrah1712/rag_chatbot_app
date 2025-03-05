import React, { useState } from "react";
import "./Chatbot.css";
import "./DeveloperConsole.css";
import CloseIcon from "@mui/icons-material/Close";
import SettingsIcon from "@mui/icons-material/Settings";
import DeveloperConsole from "./DeveloperConsole";

function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [adminKey, setAdminKey] = useState("");
  const [accessGranted, setAccessGranted] = useState(false);
  const [messages, setMessages] = useState([
    { text: "Hey there! 👋", sender: "bot" },
    { text: "How can I assist you?", sender: "bot" },
  ]);
  const [inputText, setInputText] = useState("");

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
    setShowSettings(false);
  };

  const handleSettingsClick = () => {
    setShowSettings(true);
  };

  const handleAdminSubmit = () => {
    if (adminKey === "admin123") {
      setAccessGranted(true);
      setShowSettings(false);
    } else {
      alert("Incorrect Admin Key!");
    }
  };

  const handleSendMessage = () => {
    if (inputText.trim() === "") return;
    setMessages([...messages, { text: inputText, sender: "user" }]);
    setInputText("");
  };

  const handleResetChat = () => {
    setMessages([
      { text: "Hey there! 👋", sender: "bot" },
      { text: "How can I assist you?", sender: "bot" },
    ]);
  };

  return (
    <div className="chatbot-container">
      {!isOpen && (
        <button className="chat-icon" onClick={toggleChatbot}>
          💬
        </button>
      )}

      {isOpen && (
        <div className="chatbox">
          <div className="chat-header">
            <span>Chat Assistant</span>
            <div className="icons">
              <button className="reset-button" onClick={handleResetChat}>
                Reset
              </button>
              <SettingsIcon className="icon" onClick={handleSettingsClick} />
              <CloseIcon className="icon" onClick={toggleChatbot} />
            </div>
          </div>

          {/* Chat UI and Developer Console together */}
          <div className="chat-body">
            {!accessGranted ? (
              messages.map((msg, index) => (
                <p
                  key={index}
                  className={msg.sender === "bot" ? "bot-message" : "user-message"}
                >
                  {msg.text}
                </p>
              ))
            ) : (
              <DeveloperConsole closeConsole={() => setAccessGranted(false)} />
            )}
          </div>

          {!accessGranted && (
            <div className="chat-input-container">
              <input
                type="text"
                className="chat-input"
                placeholder="Type your message..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
              />
              <button className="send-button" onClick={handleSendMessage}>
                Send
              </button>
            </div>
          )}

          {showSettings && (
            <div className="admin-modal">
              <h3>Enter Admin Key</h3>
              <input
                type="password"
                value={adminKey}
                onChange={(e) => setAdminKey(e.target.value)}
                placeholder="Enter admin key"
              />
              <button onClick={handleAdminSubmit}>Submit</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Chatbot;
