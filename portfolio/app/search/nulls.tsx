'use client'

import { useState, useEffect, useRef } from 'react'
import { Search, Loader2, ExternalLink, Sparkles, MessageCircle, Trash2 } from 'lucide-react'

interface SearchResult {
  title: string
  url: string
  snippet: string
  rank: number
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ChatWithSearchResponse {
  ai_response: string
  search_performed: boolean
  search_query?: string
  sources_used: SearchResult[]
  response_time: number
}

export default function SearchPage() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lastSearchInfo, setLastSearchInfo] = useState<{
    performed: boolean
    query?: string
    sources: SearchResult[]
    responseTime: number
  } | null>(null)

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
  }, [messages, loading])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = input.trim()
    setInput('')
    setLoading(true)
    setError('')

    // Add user message to conversation
    const newMessages = [...messages, { role: 'user' as const, content: userMessage }]
    setMessages(newMessages)

    try {
      const response = await fetch('/api/chat-search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          messages: newMessages,
          search_when_needed: true,
          num_search_results: 5
        }),
      })

      if (!response.ok) {
        throw new Error('Chat failed')
      }

      const data: ChatWithSearchResponse = await response.json()
      
      // Add assistant response to conversation
      setMessages([...newMessages, { role: 'assistant', content: data.ai_response }])
      
      // Store search info for display
      setLastSearchInfo({
        performed: data.search_performed,
        query: data.search_query,
        sources: data.sources_used,
        responseTime: data.response_time
      })

    } catch (err) {
      setError('Failed to send message. Please try again.')
      console.error('Chat error:', err)
      // Remove the user message if the request failed
      setMessages(messages)
    } finally {
      setLoading(false)
    }
  }

  const clearChat = () => {
    setMessages([])
    setLastSearchInfo(null)
    setError('')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <MessageCircle className="w-8 h-8 text-purple-400" />
            <h1 className="text-4xl font-bold text-white">Search with kh.AI</h1>
          </div>
          <p className="text-gray-300 text-lg">
            Ask and I will search.
          </p>
        </div>

        {/* Chat Container */}
        <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl p-6 mb-6">
          {/* Chat History - THIS is the only scrollable area */}
          <div 
            ref={chatContainerRef}
            className="h-96 overflow-y-auto mb-4 space-y-4 scroll-smooth"
            style={{ scrollBehavior: 'smooth' }}
          >
            {messages.length === 0 ? (
              <div className="text-gray-400 text-center py-16">
                <MessageCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Start a conversation...</p>
                <p className="text-sm mt-2">Try asking: "Who is the current president?" or "What's 2+2?"</p>
              </div>
            ) : (
              <>
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] p-4 rounded-lg ${
                        message.role === 'user'
                          ? 'bg-purple-600 text-white'
                          : 'bg-white/10 text-gray-200 border border-white/20'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start">
                    <div className="bg-white/10 border border-white/20 rounded-lg p-4">
                      <div className="flex items-center gap-2 text-gray-300">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Thinking and searching when needed...</span>
                      </div>
                    </div>
                  </div>
                )}
                {/* This div helps ensure we can scroll to the bottom */}
                <div ref={messagesEndRef} style={{ height: '1px' }} />
              </>
            )}
          </div>

          {/* Search Status */}
          {lastSearchInfo && messages.length > 0 && (
            <div className="mb-4 p-3 bg-white/5 border border-white/10 rounded-lg">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2 text-gray-400">
                  {lastSearchInfo.performed ? (
                    <>
                      <Sparkles className="w-4 h-4 text-green-400" />
                      <span>Searched for current information</span>
                      {lastSearchInfo.query && (
                        <span className="text-purple-300">"{lastSearchInfo.query}"</span>
                      )}
                    </>
                  ) : (
                    <>
                      <MessageCircle className="w-4 h-4 text-blue-400" />
                      <span>Used existing knowledge</span>
                    </>
                  )}
                </div>
                <span className="text-gray-500">
                  {lastSearchInfo.responseTime.toFixed(2)}s
                </span>
              </div>
            </div>
          )}

          {/* Input Form */}
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg transition-colors"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 text-white animate-spin" />
              ) : (
                <Search className="w-5 h-5 text-white" />
              )}
            </button>
            {messages.length > 0 && (
              <button
                type="button"
                onClick={clearChat}
                className="px-4 py-3 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                title="Clear conversation"
              >
                <Trash2 className="w-5 h-5 text-white" />
              </button>
            )}
          </form>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-red-200 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Sources section remains the same */}
        {lastSearchInfo?.performed && lastSearchInfo.sources.length > 0 && (
          <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <ExternalLink className="w-5 h-5" />
              Sources Used ({lastSearchInfo.sources.length})
            </h3>
            <div className="space-y-4">
              {lastSearchInfo.sources.map((source, index) => (
                <div
                  key={index}
                  className="border border-white/10 rounded-lg p-4 hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 bg-purple-600 text-white text-sm rounded-full flex items-center justify-center font-medium">
                      {index + 1}
                    </span>
                    <div className="flex-1 min-w-0">
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-purple-300 hover:text-purple-200 font-medium transition-colors block"
                      >
                        {source.title}
                      </a>
                      {source.snippet && (
                        <p className="text-gray-400 text-sm mt-1 line-clamp-2">
                          {source.snippet}
                        </p>
                      )}
                      <p className="text-gray-500 text-xs mt-1 truncate">
                        {source.url}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}