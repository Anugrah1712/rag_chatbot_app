// DeveloperConsole.js
import React, { useState } from "react";
import "./DeveloperConsole.css";

const DeveloperConsole = ({ closeConsole, openChatbot }) => {
  // State variables
  const [documents, setDocuments] = useState([]);
  const [urls, setUrls] = useState([""]);
  const [chunkSize, setChunkSize] = useState(1500);
  const [chunkOverlap, setChunkOverlap] = useState(500);
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPreprocessed, setIsPreprocessed] = useState(false);
  const [selectedVectorDB, setSelectedVectorDB] = useState("");
  const [selectedChatModel, setSelectedChatModel] = useState("");

  const embeddingModels = [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2",
    "distilbert-base-nli-stsb-mean-tokens",
    "all-distilroberta-v1",
    "paraphrase-distilroberta-base-v1",
    "stsb-roberta-base",
  ];

  const vectorDBs = ["Chroma", "FAISS" , "Pinecone" ,"Weaviate" , "Qdrant"];

  const chatModels = [
    "Qwen/QwQ-32B-Preview",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "scb10x/scb10x-llama3-typhoon-v1-5-8b-instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  ];

  const handleDocumentUpload = (event) => {
    const files = Array.from(event.target.files);
    setDocuments([...documents, ...files]);
  };

  const removeDocument = (index) => {
    const updatedDocs = documents.filter((_, i) => i !== index);
    setDocuments(updatedDocs);
  };

  const handleUrlChange = (index, value) => {
    const newUrls = [...urls];
    newUrls[index] = value;
    setUrls(newUrls);
  };

  const removeUrl = (index) => {
    const updatedUrls = urls.filter((_, i) => i !== index);
    setUrls(updatedUrls);
  };

  const addUrlField = () => setUrls([...urls, ""]);

  const handlePreprocessing = async () => {
    if (!selectedEmbeddingModel) {
      alert("Please select an embedding model.");
      return;
    }

    setIsProcessing(true);

    const formData = new FormData();
    documents.forEach((doc) => formData.append("doc_files", doc));
    formData.append("links", JSON.stringify(urls.filter((url) => url.trim() !== "")));
    formData.append("embedding_model", selectedEmbeddingModel);
    formData.append("chunk_size", chunkSize);
    formData.append("chunk_overlap", chunkOverlap);

    try {
      console.log("Sending preprocessing request...");

      const response = await fetch("http://localhost:8000/preprocess", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      console.log("Preprocessing Response:", result);

      if (response.ok || result.status === "success") {
        setIsPreprocessed(true);
        alert("✅ Preprocessing completed successfully!");
      } else {
        throw new Error(result.detail || "Preprocessing failed");
      }
    } catch (error) {
      console.error("❌ Preprocessing error:", error);
      alert("❌ Preprocessing failed. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFinalConfiguration = async () => {
    if (!selectedVectorDB || !selectedChatModel) {
      console.error("Please select both vector database and chat model.");
      return;
    }
  
    try {
      // Select vector database
      const vectordbResponse = await fetch("http://localhost:8000/select_vectordb", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ vectordb: selectedVectorDB }),
      });
  
      if (!vectordbResponse.ok) throw new Error("Vector database selection failed.");
      await vectordbResponse.json();
  
      // Select chat model
      const chatModelResponse = await fetch("http://localhost:8000/select_chat_model", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ chat_model: selectedChatModel }),
      });
  
      if (!chatModelResponse.ok) throw new Error("Chat model selection failed.");
      await chatModelResponse.json();
  
      alert("✅ Chatbot is ready!");
  
      // Close Developer Console
      if (closeConsole) closeConsole();
  
      // Open Chatbot (Check if function exists before calling)
      if (openChatbot && typeof openChatbot === "function") {
        openChatbot();
      } else {
        console.error("❌ openChatbot function is undefined or not a function.");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("❌ Error in configuration. Please try again.");
    }
  };
  

  return (
    <div className="developer-console">
      <h2>Developer Console</h2>

      {/* Step 1: Upload Docs, URLs, Select Embeddings */}
      {!isPreprocessed && (
        <>
          <div className="section">
            <label>Upload Documents (PDF/DOCX):</label>
            <input type="file" multiple accept=".pdf,.docx" onChange={handleDocumentUpload} />
            <ul>
              {documents.map((doc, index) => (
                <li key={index}>
                  {doc.name}
                  <button className="remove-btn" onClick={() => removeDocument(index)}>❌</button>
                </li>
              ))}
            </ul>
          </div>

          <div className="section">
            <label>Enter URLs for Web Scraping:</label>
            {urls.map((url, index) => (
              <div key={index} className="url-field">
                <input
                  type="text"
                  placeholder={`URL ${index + 1}`}
                  value={url}
                  onChange={(e) => handleUrlChange(index, e.target.value)}
                />

                <button className="remove-btn" onClick={() => removeUrl(index)}>❌</button>
              </div>
            ))}
            <button onClick={addUrlField}>+ Add More</button>
          </div>

          <div className="section">
            <label>Chunk Size:</label>
            <input type="number" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} />
          </div>

          <div className="section">
            <label>Chunk Overlap:</label>
            <input type="number" value={chunkOverlap} onChange={(e) => setChunkOverlap(Number(e.target.value))} />
          </div>

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

          <button onClick={handlePreprocessing} disabled={isProcessing}>
            {isProcessing ? "Processing..." : "Start Preprocessing"}
          </button>
        </>
      )}

      {/* Step 2: Select Vector DB & Chat Model (Only show after preprocessing) */}
      {isPreprocessed && (
        <>
          <h3>✅ Preprocessing completed!</h3>

          <div className="section">
            <label>Select Vector Database:</label>
            <select value={selectedVectorDB} onChange={(e) => setSelectedVectorDB(e.target.value)}>
              <option value="">Select Database</option>
              {vectorDBs.map((db) => (
                <option key={db} value={db}>
                  {db}
                </option>
              ))}
            </select>
          </div>

          <div className="section">
            <label>Select Chat Model:</label>
            <select value={selectedChatModel} onChange={(e) => setSelectedChatModel(e.target.value)}>
              <option value="">Select Chat Model</option>
              {chatModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>

          <button onClick={handleFinalConfiguration}>Run Chatbot</button>
        </>
      )}
    </div>
  );
};

export default DeveloperConsole;