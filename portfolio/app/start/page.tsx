"use client";

import { useState, useEffect, useRef } from "react";

type ChatMessage = { role: "user" | "assistant"; content: string };

interface EnhancedUploadResponse {
  result: string;
  session_id: string;
  filename: string;
  message: string;
  total_pages: number;
  total_chunks: number;
  document_size: string;
  processing_method: string;
}

export default function UploadPage() {
  const [apiUrl, setApiUrl] = useState<string>("");
  
  // PDF upload states
  const [file, setFile] = useState<File | null>(null);
  const [pdfOutput, setPdfOutput] = useState<string>("");
  const [pdfError, setPdfError] = useState<string>("");
  const [pdfLoading, setPdfLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{
    stage: string;
    details: string;
  } | null>(null);

  // Chat states
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string>("");

  // Session state - Enhanced for mid-conversation uploads
  const [sessionId, setSessionId] = useState<string>("");
  const [pdfFilename, setPdfFilename] = useState<string>("");
  const [availableSessions, setAvailableSessions] = useState<string[]>([]);

  // Ref for auto-scroll - ONLY for chat container
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom - ONLY within chat container
  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      // Scroll the chat container to the bottom
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }

  useEffect(() => {
    // Small delay to ensure DOM is updated
    const timer = setTimeout(() => {
      scrollToBottom()
    }, 100)
    
    return () => clearTimeout(timer)
  }, [chatMessages, chatLoading])

  // Set API URL after component mounts
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const url = window.location.hostname === 'localhost' 
        ? "http://localhost:8000" 
        : "https://goldfish-app-84zag.ondigitalocean.app/my-portfolio-portfolio-api";
      setApiUrl(url);
    }
  }, []);

  // Add file size constants
  const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
  
  // Update file change handler with validation
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] ?? null;
    setFile(selectedFile);
    setPdfError(""); // Clear any previous errors
    
    if (selectedFile) {
      const sizeMB = (selectedFile.size / (1024 * 1024)).toFixed(1);
      console.log(`Selected file: ${selectedFile.name} (${sizeMB}MB)`);
      
      if (selectedFile.size > MAX_FILE_SIZE) {
        setPdfError(`File too large (${sizeMB}MB). Maximum size is 50MB. Please compress your PDF or use a smaller file.`);
      }
    }
  };

  // Enhanced upload handler with better error messages
  const handlePdfSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || pdfLoading || !apiUrl) return;
    
    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
    console.log(`Uploading file: ${file.name}, Size: ${fileSizeMB}MB`);
    
    setPdfError(""); 
    setPdfOutput(""); 
    setPdfLoading(true);
    setUploadProgress({ stage: "Uploading...", details: `Sending ${fileSizeMB}MB file to server` });
    
    try {
      const form = new FormData();
      form.append("file", file);
      
      const res = await fetch(`${apiUrl}/upload`, { 
        method: "POST", 
        body: form
      });
      
      // Add this check for non-200 responses
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server error (${res.status}): ${errorText}`);
      }
      
      const data = await res.json();
      console.log('Response data:', data);
      
      // Check if response contains an error
      if (data.error) {
        let errorMessage = data.error;
        if (data.details) {
          errorMessage += `\n\nDetails: ${data.details}`;
        }
        throw new Error(errorMessage);
      }
      
      // Check if it's a successful response with session_id
      if (!data.session_id) {
        console.error('Received data:', data); // Debug log
        throw new Error(`Invalid response from server. Expected session_id but got: ${JSON.stringify(data)}`);
      }
      
      setUploadProgress({ stage: "Processing...", details: "Creating embeddings and summary" });
      
      // Success - show processing results
      const processingInfo = `✅ Processed ${data.total_chunks} chunks from ${data.total_pages} pages using ${data.processing_method}`;
      
      if (chatMessages.length > 0) {
        const uploadNotification: ChatMessage = {
          role: "assistant",
          content: `📄 ${data.message}\n\n${processingInfo}\n\n**Document Summary:**\n${data.result}`
        };
        setChatMessages(prev => [...prev, uploadNotification]);
      } else {
        setPdfOutput(`${processingInfo}\n\n${data.result}`);
      }
      
      setSessionId(data.session_id);
      setPdfFilename(data.filename);
      setFile(null);
      
    } catch (err: any) {
      console.error('Upload error:', err);
      setPdfError(err.message || 'Upload failed. Please try again.');
    } finally {
      setPdfLoading(false);
      setUploadProgress(null);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = chatInput.trim();
    if (!text || chatLoading) return;

    const newUserMsg: ChatMessage = { role: "user", content: text };
    const updatedHistory = [...chatMessages, newUserMsg];
    setChatMessages(updatedHistory);
    setChatInput("");
    setChatLoading(true);
    setChatError("");

    try {
      const res = await fetch(`${apiUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          messages: updatedHistory,
          session_id: sessionId // Send current session_id if available
        }),
      });
      const data = await res.json();
      
      if (data.error) {
        setChatError(data.error);
      } else if (data.result) {
        const assistantMsg: ChatMessage = { role: "assistant", content: data.result };
        setChatMessages([...updatedHistory, assistantMsg]);
      } else {
        setChatError("No response from server.");
      }
    } catch (err: any) {
      setChatError(err.message);
    } finally {
      setChatLoading(false);
    }
  };

  // Function to start fresh conversation
  const startFreshConversation = () => {
    setChatMessages([]);
    setSessionId("");
    setPdfFilename("");
    setPdfOutput("");
  };

  return (
    <div className="min-h-screen bg-black p-8 flex flex-col items-center">
      {/* Layout: PDF Upload Card on the left, Chat on the right */}
      <div className="w-full max-w-5xl flex flex-col md:flex-row gap-8 mb-8">
        {/* PDF Upload Card */}
        <div className="w-full md:w-1/3 max-w-md bg-white rounded-xl shadow-lg p-6 mb-6 md:mb-0">
          <div className="flex justify-between items-center mb-2">
            <h1 className="text-lg font-bold text-black">Upload PDF</h1>
            {chatMessages.length > 0 && (
              <button
                onClick={startFreshConversation}
                className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700"
              >
                Start Fresh
              </button>
            )}
          </div>

        <form onSubmit={handlePdfSubmit} className="flex justify-between items-center mb-2">
            <label className="border-2 border-dashed rounded-lg p-2 text-center hover:border-blue-400 cursor-pointer bg-gray-100 text-black text-xs w-40 mx-auto">
            <input
              type="file"
              accept="application/pdf"
              className="hidden"
              disabled={pdfLoading}
              onChange={handleFileChange}
            />
        {file ? (
          <div>
            <p className="font-medium truncate">{file.name}</p>
            <p className="text-xs text-gray-600 mt-1">
          {(file.size / (1024 * 1024)).toFixed(1)}MB
            </p>
          </div>
        ) : (
          <div>
            <p>Click to select a PDF</p>
            <p className="text-xs text-gray-600 mt-1">Max:10MB</p>
          </div>
        )}
          </label>
          <button
        type="submit"
        disabled={pdfLoading || !file || (file && file.size > MAX_FILE_SIZE)}
        className={`inline-flex items-center justify-center gap-2 px-6 py-2 rounded-lg text-white font-semibold text-sm shadow transition ${
          pdfLoading || (file && file.size > MAX_FILE_SIZE)
            ? "bg-gray-400 cursor-not-allowed opacity-80"
            : "bg-gradient-to-r from-gray-900 to-gray-900 hover:from-blue-700 hover:to-blue-900 cursor-pointer"
        }`}
        style={{ alignSelf: "center", width: "auto" }}
          >
        {pdfLoading ? "Processing…" : chatMessages.length > 0 ? "Add PDF" : "Upload"}
          </button>
        </form>
        
        {pdfError && (
          <div className="mt-2 p-2 bg-red-100 border border-red-300 rounded">
        <p className="text-red-700 text-xs">{pdfError}</p>
        {pdfError.includes('too large') && (
          <div className="mt-1 text-red-600 text-xs">
            <p>💡 Reduce PDF size:</p>
            <ul className="list-disc list-inside mt-1 space-y-0.5">
          <li>Use online PDF compressors</li>
          <li>Split large documents</li>
          <li>Remove unnecessary images/pages</li>
          <li>Save with lower quality</li>
            </ul>
          </div>
        )}
          </div>
        )}
        
        {pdfOutput && chatMessages.length === 0 && (
          <div className="mt-2">
        {/* <pre className="p-2 bg-gray-900 text-white rounded text-xs whitespace-pre-wrap">
          {pdfOutput}
        </pre> */}
        {sessionId && (
          <div className="mt-1 p-1 bg-green-100 border border-green-300 rounded text-green-800 text-xs">
            ✓ PDF "{pdfFilename}" is ready for chat.
          </div>
        )}
          </div>
        )}

        {uploadProgress && (
          <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded">
        <div className="flex items-center gap-2">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          <div>
            <p className="font-medium text-blue-800 text-xs">{uploadProgress.stage}</p>
            <p className="text-xs text-blue-600">{uploadProgress.details}</p>
          </div>
        </div>
          </div>
        )}
      </div>

      {/* Chat Bot Card */}
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-lg p-8">
        <h2 className="text-2xl font-bold mb-4 text-black">
          Chat {sessionId && pdfFilename && (
            <span className="text-sm font-normal text-gray-600">
              (current context: {pdfFilename})
            </span>
          )}
        </h2>
        
        <div 
          ref={chatContainerRef}
          className="h-64 overflow-y-auto mb-4 space-y-2 text-black scroll-smooth"
          style={{ scrollBehavior: 'smooth' }}
        >
          {chatMessages.length === 0 ? (
            <div className="text-gray-500 text-center py-8">
              Start a conversation or upload a PDF to begin...
            </div>
          ) : (
            <>
              {chatMessages.map((m, i) => (
                <div
                  key={i}
                  className={`p-3 rounded-lg max-w-[80%] ${
                    m.role === "user" 
                      ? "bg-blue-500 text-white ml-auto" 
                      : "bg-gray-200 text-black mr-auto"
                  }`}
                >
                  <div className="whitespace-pre-wrap">{m.content}</div>
                </div>
              ))}
              {chatLoading && (
                <div className="bg-gray-200 text-black mr-auto p-3 rounded-lg max-w-[80%]">
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                    <span>Thinking...</span>
                  </div>
                </div>
              )}
              {/* Invisible div for auto-scroll target */}
              <div ref={messagesEndRef} style={{ height: '1px' }} />
            </>
          )}
        </div>
        
        <form onSubmit={handleChatSubmit} className="flex gap-2 text-black">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            disabled={chatLoading}
            placeholder={sessionId ? "Ask about your PDF or anything else..." : "Ask a question..."}
            className="flex-1 border rounded px-3 py-2 text-black"
          />
          <button
            type="submit"
            disabled={chatLoading}
            className={`px-4 py-2 rounded-lg text-white font-semibold ${
              chatLoading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-gray-900 hover:bg-blue-800 cursor-pointer"
            }`}
          >
            {chatLoading ? "…" : "Send"}
          </button>
        </form>
        {chatError && <p className="mt-2 text-red-600">{chatError}</p>}
      </div>
      </div>
    </div>
  );
}
