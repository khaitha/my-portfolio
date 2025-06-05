"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://goldfish-app-84zag.ondigitalocean.app/my-portfolio-portfolio-api";

type ChatMessage = { role: "user" | "assistant"; content: string };

export default function UploadPage() {
  // PDF upload states
  const [file, setFile] = useState<File | null>(null);
  const [pdfOutput, setPdfOutput] = useState<string>("");
  const [pdfError, setPdfError] = useState<string>("");
  const [pdfLoading, setPdfLoading] = useState(false);

  // Chat states
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string>("");

  // Session state - Enhanced for mid-conversation uploads
  const [sessionId, setSessionId] = useState<string>("");
  const [pdfFilename, setPdfFilename] = useState<string>("");
  const [availableSessions, setAvailableSessions] = useState<string[]>([]);

  const handlePdfSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || pdfLoading) return;
    setPdfError(""); 
    setPdfOutput(""); 
    setPdfLoading(true);
    
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
      const data = await res.json();
      
      if (data.error) {
        setPdfError(data.error);
      } else {
        // Add upload notification to chat if we have existing conversation
        if (chatMessages.length > 0) {
          const uploadNotification: ChatMessage = {
            role: "assistant",
            content: `📄 ${data.message}\n\n**Document Summary:**\n${data.result}`
          };
          setChatMessages(prev => [...prev, uploadNotification]);
        } else {
          // If no existing conversation, show in PDF output area
          setPdfOutput(data.result);
        }
        
        setSessionId(data.session_id);
        setPdfFilename(data.filename);
        setAvailableSessions(prev => [...prev, data.session_id]);
        console.log("Session ID:", data.session_id);
        
        // Don't clear chat history - maintain conversation!
        // setChatMessages([]); // REMOVED - this was clearing the conversation
      }
    } catch (err: any) {
      setPdfError(err.message);
    } finally {
      setPdfLoading(false);
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
      const res = await fetch(`${API_URL}/chat`, {
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
      {/* PDF Upload Card */}
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-lg p-8 mb-8">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold text-black">Upload PDF</h1>
          {chatMessages.length > 0 && (
            <button
              onClick={startFreshConversation}
              className="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 rounded-lg text-gray-700"
            >
              Start Fresh
            </button>
          )}
        </div>
        
        <form onSubmit={handlePdfSubmit} className="flex flex-col gap-4">
          <label className="border-2 border-dashed rounded-xl p-6 text-center hover:border-blue-400 cursor-pointer bg-gray-100 text-black">
            <input
              type="file"
              accept="application/pdf"
              className="hidden"
              disabled={pdfLoading}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            {file ? file.name : "Click to select a PDF"}
          </label>
          <button
            type="submit"
            disabled={pdfLoading}
            className={`inline-flex items-center justify-center gap-2 px-10 py-3 rounded-lg text-white font-semibold text-base shadow transition ${
              pdfLoading
                ? "bg-gray-400 cursor-not-allowed opacity-80"
                : "bg-gradient-to-r from-gray-900 to-gray-900 hover:from-blue-700 hover:to-blue-900 cursor-pointer"
            }`}
            style={{ alignSelf: "center", width: "auto" }}
          >
            {pdfLoading ? "Processing…" : chatMessages.length > 0 ? "Add PDF to Conversation" : "Upload & Analyze"}
          </button>
        </form>
        
        {pdfError && <p className="mt-2 text-red-600">{pdfError}</p>}
        
        {/* Only show PDF output if no conversation exists */}
        {pdfOutput && chatMessages.length === 0 && (
          <div className="mt-4">
            <pre className="p-4 bg-gray-900 text-white rounded whitespace-pre-wrap">
              {pdfOutput}
            </pre>
            {sessionId && (
              <div className="mt-2 p-2 bg-green-100 border border-green-300 rounded text-green-800 text-sm">
                ✓ PDF "{pdfFilename}" is ready for chat. Ask questions about the document below!
              </div>
            )}
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
        
        <div className="h-64 overflow-y-auto mb-4 space-y-2 text-black">
          {chatMessages.length === 0 ? (
            <div className="text-gray-500 text-center py-8">
              Start a conversation or upload a PDF to begin...
            </div>
          ) : (
            chatMessages.map((m, i) => (
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
            ))
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
  );
}
