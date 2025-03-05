import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./DeveloperConsole.css";

const DeveloperConsole = () => {
  const navigate = useNavigate();

  // State variables
  const [documents, setDocuments] = useState([]);
  const [urls, setUrls] = useState([""]);
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState("");
  const [chunkSize, setChunkSize] = useState(2000);
  const [chunkOverlap, setChunkOverlap] = useState(500);
  const [selectedVectorDB, setSelectedVectorDB] = useState("");
  const [selectedChatModel, setSelectedChatModel] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  // Embedding models
  const embeddingModels = [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2",
    "distilbert-base-nli-stsb-mean-tokens",
    "all-distilroberta-v1",
    "paraphrase-distilroberta-base-v1",
    "stsb-roberta-base",
  ];

  // Vector databases
  const vectorDBs = ["Pinecone", "Chroma", "FAISS", "Qdrant", "Weaviate"];

  // Chat models
  const chatModels = [
    "Qwen/QwQ-32B-Preview",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "scb10x/scb10x-llama3-typhoon-v1-5-8b-instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  ];

  // Handle document upload
  const handleDocumentUpload = (event) => {
    const files = Array.from(event.target.files);
    setDocuments([...documents, ...files]);
  };

  // Handle URL input change
  const handleUrlChange = (index, value) => {
    const newUrls = [...urls];
    newUrls[index] = value;
    setUrls(newUrls);
  };

  // Add more URL fields
  const addUrlField = () => {
    setUrls([...urls, ""]);
  };

  // Handle sending configurations to the API
  const handleSubmitConfigurations = async () => {
    if (!selectedEmbeddingModel || !selectedVectorDB || !selectedChatModel) {
      alert("Please select all required options: embedding model, vector database, and chat model.");
      return;
    }

    setIsProcessing(true);

    // Prepare form data for file upload
    const formData = new FormData();
    formData.append("embedding_model", selectedEmbeddingModel);
    formData.append("chat_model", selectedChatModel);
    formData.append("vector_database", selectedVectorDB);
    formData.append("chunk_size", chunkSize);
    formData.append("chunk_overlap", chunkOverlap);

    urls.forEach((url, index) => {
      if (url.trim() !== "") {
        formData.append(`web_links`, url);
      }
    });

    documents.forEach((doc) => {
      formData.append("documents", doc);
    });

    try {
      // Send configurations and files
      const configResponse = await fetch("http://localhost:8000/api/set_config", {
        method: "POST",
        body: formData,
      });

      if (!configResponse.ok) {
        throw new Error("Error updating configurations.");
      }

      // Trigger preprocessing in the background
      const setupResponse = await fetch("http://localhost:8000/api/setup_chatbot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          embedding_model: selectedEmbeddingModel,
          chat_model: selectedChatModel,
          vector_database: selectedVectorDB,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
          web_links: urls.filter((url) => url.trim() !== ""),
        }),
      });

      if (setupResponse.ok) {
        alert("Configurations sent successfully! Chatbot is now ready.");
        navigate("/chatbot"); // Redirect to chatbot
      } else {
        throw new Error("Error setting up chatbot.");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="developer-console">
      <h2>Developer Console</h2>

      {/* Document Upload */}
      <div className="section">
        <label>Upload Documents (PDF/DOCX):</label>
        <input type="file" multiple accept=".pdf,.docx" onChange={handleDocumentUpload} />
        <ul>
          {documents.map((doc, index) => (
            <li key={index}>{doc.name}</li>
          ))}
        </ul>
      </div>

      {/* URL Inputs */}
      <div className="section">
        <label>Enter URLs for Web Scraping:</label>
        {urls.map((url, index) => (
          <input
            key={index}
            type="text"
            placeholder={`URL ${index + 1}`}
            value={url}
            onChange={(e) => handleUrlChange(index, e.target.value)}
          />
        ))}
        <button onClick={addUrlField}>+ Add More</button>
      </div>

      {/* Embedding Model Selection */}
      <div className="section">
        <label>Select Embedding Model:</label>
        <select value={selectedEmbeddingModel} onChange={(e) => setSelectedEmbeddingModel(e.target.value)}>
          <option value="">Select Model</option>
          {embeddingModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>

      {/* Chunk Settings */}
      <div className="section">
        <label>Chunk Size:</label>
        <input type="number" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} />
      </div>
      <div className="section">
        <label>Chunk Overlap:</label>
        <input type="number" value={chunkOverlap} onChange={(e) => setChunkOverlap(Number(e.target.value))} />
      </div>

      {/* Vector Database Selection */}
      <div className="section">
        <label>Select Vector Database:</label>
        <select value={selectedVectorDB} onChange={(e) => setSelectedVectorDB(e.target.value)}>
          <option value="">Select DB</option>
          {vectorDBs.map((db) => (
            <option key={db} value={db}>
              {db}
            </option>
          ))}
        </select>
      </div>

      {/* Chat Model Selection */}
      <div className="section">
        <label>Select Chat Model:</label>
        <select value={selectedChatModel} onChange={(e) => setSelectedChatModel(e.target.value)}>
          <option value="">Select Model</option>
          {chatModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>

      {/* Submit Configurations */}
      <button onClick={handleSubmitConfigurations} disabled={isProcessing}>
        {isProcessing ? "Processing..." : "Submit & Start Chatbot"}
      </button>
    </div>
  );
};

export default DeveloperConsole;
