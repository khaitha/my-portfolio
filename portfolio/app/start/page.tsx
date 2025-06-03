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

  const handlePdfSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || pdfLoading) return;
    setPdfError(""); setPdfOutput(""); setPdfLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (data.error) setPdfError(data.error);
      else setPdfOutput(data.result);
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
        body: JSON.stringify({ messages: updatedHistory }),
      });
      const data = await res.json();
      if (data.result) {
        setChatMessages((h) => [...h, { role: "assistant", content: data.result }]);
      } else {
        setChatError("No response from server.");
      }
    } catch (err: any) {
      setChatError(err.message);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black p-8 flex flex-col items-center">
      {/* PDF Upload Card */}
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-lg p-8 mb-8">
        <h1 className="text-2xl font-bold mb-4 text-black">Upload PDF</h1>
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
            {pdfLoading ? "Processing…" : "Upload & Analyze"}
          </button>
        </form>
        {pdfError && <p className="mt-2 text-red-600">{pdfError}</p>}
        {pdfOutput && (
          <pre className="mt-4 p-4 bg-gray-900 text-white rounded whitespace-pre-wrap">
            {pdfOutput}
          </pre>
        )}
      </div>

      {/* Chat Bot Card */}
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-lg p-8">
        <h2 className="text-2xl font-bold mb-4 text-black">Chat</h2>
        <div className="h-64 overflow-y-auto mb-4 space-y-2 text-black">
          {chatMessages.map((m, i) => (
            <div
              key={i}
              className={`p-2 rounded ${
                m.role === "user" ? "bg-blue-500 self-end" : "bg-green-500 self-start"
              }`}
            >
              {m.content}
            </div>
          ))}
        </div>
        <form onSubmit={handleChatSubmit} className="flex gap-2 text-black">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            disabled={chatLoading}
            placeholder="Ask a question…"
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
