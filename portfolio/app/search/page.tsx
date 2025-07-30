'use client'

import { useState, useEffect, useRef } from 'react'
import { Search, Loader2, ExternalLink, MessageCircle, Trash2, Code, Play, Terminal, Edit3, X, Copy, Check, AlertCircle, RotateCcw, Upload, File, FileText } from 'lucide-react'

declare global {
  interface Window {
    pyodide: any;
    loadPyodide: () => Promise<any>;
  }
}

interface SearchResult {
  title: string
  url: string
  snippet: string
  rank: number
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  hasCode?: boolean
  codeId?: string
  metadata?: {
    searchPerformed?: boolean
    executionTime?: number
  }
}

interface ChatWithSearchResponse {
  ai_response: string
  search_performed: boolean
  search_query?: string
  sources_used: SearchResult[]
  response_time: number
}

interface ExecutionResult {
  success: boolean
  output?: string
  error?: string
  executionTime?: number
}

interface CodeSession {
  id: string
  code: string
  result: ExecutionResult | null
  timestamp: number
  description: string
  requiredPackages: string[]
  versions: CodeVersion[]  // Track code versions
}

interface CodeVersion {
  code: string
  timestamp: number
  description: string
  requiredPackages: string[]
}

interface CodeActionAnalysis {
  action: 'generate' | 'edit' | 'execute' | 'revert' | 'question' | 'none'
  confidence: number
  context?: {
    isPlottingRelated?: boolean
    hasCodeContext?: boolean
    needsExecution?: boolean
    versionRequest?: {
      type: 'previous' | 'specific'
      steps?: number
    }
  }
}

// PDF-related interfaces
interface UploadResponse {
  result: string
  session_id: string
  filename: string
  message: string
  is_mid_conversation: boolean
  total_pages?: number
  total_chunks?: number
  document_size?: string
  processing_method?: string
}

interface PDFContext {
  sessionId: string
  filename: string
  summary: string
  isActive: boolean
  uploadedAt: number
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

  // Code execution states
  const [apiUrl, setApiUrl] = useState<string>("")
  const [pyodideReady, setPyodideReady] = useState(false)
  const [pyodideLoading, setPyodideLoading] = useState(false)
  const [executing, setExecuting] = useState(false)
  const [codeSessionActive, setCodeSessionActive] = useState(false)
  const [codeSessions, setCodeSessions] = useState<CodeSession[]>([])
  const [activeCodeId, setActiveCodeId] = useState<string | null>(null)
  const [editingCode, setEditingCode] = useState<string>('')
  const [loadedPackages, setLoadedPackages] = useState<Set<string>>(new Set())
  const [copiedCode, setCopiedCode] = useState<string | null>(null)
  const [executionProgress, setExecutionProgress] = useState<number>(0)

  // PDF upload states
  const [pdfContext, setPdfContext] = useState<PDFContext | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')

  // Refs for auto-scroll
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }

  useEffect(() => {
    const timer = setTimeout(scrollToBottom, 100)
    return () => clearTimeout(timer)
  }, [messages, loading])

  // Function to detect potential infinite loops
  const detectPotentialInfiniteLoops = (code: string): string[] => {
    const warnings: string[] = []
    const lines = code.toLowerCase().split('\n')
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim()
      
      // Check for while True without break
      if (line.includes('while true') || line.includes('while 1')) {
        const hasBreak = lines.slice(i, Math.min(i + 10, lines.length))
          .some(l => l.includes('break') || l.includes('return'))
        if (!hasBreak) {
          warnings.push(`Line ${i + 1}: 'while True' without visible break condition`)
        }
      }
      
      // Check for while loops with conditions that may never change
      if (line.startsWith('while ') && !line.includes('input') && !line.includes('random')) {
        const hasBreak = lines.slice(i, Math.min(i + 10, lines.length))
          .some(l => l.includes('break') || l.includes('return'))
        if (!hasBreak) {
          warnings.push(`Line ${i + 1}: while loop without visible break condition`)
        }
      }
      
      // Check for for loops with very large ranges
      const forMatch = line.match(/for.*range\s*\(\s*(\d+)\s*\)/)
      if (forMatch) {
        const range = parseInt(forMatch[1])
        if (range > 100000) {
          warnings.push(`Line ${i + 1}: Large range(${range}) may cause browser freeze`)
        }
      }
      
      // Check for nested loops that could be problematic
      if ((line.includes('for ') || line.includes('while ')) && i > 0) {
        const prevLines = lines.slice(Math.max(0, i-5), i)
        const hasNestedLoop = prevLines.some(l => l.includes('for ') || l.includes('while '))
        if (hasNestedLoop) {
          warnings.push(`Line ${i + 1}: Nested loops detected - may cause performance issues`)
        }
      }
    }
    
    return warnings
  }

  // Function to copy code to clipboard
  const copyCodeToClipboard = async (code: string) => {
    try {
      await navigator.clipboard.writeText(code)
      setCopiedCode(code)
      setTimeout(() => setCopiedCode(null), 2000)
    } catch (error) {
      console.error('Failed to copy code:', error)
    }
  }

  // Function to move matplotlib canvases to our output container
  const moveMatplotlibCanvases = () => {
    setTimeout(() => {
      const container = document.getElementById('matplotlib-output-container')
      if (!container) return

      // Find all matplotlib canvases (but not rubberband)
      const canvases = document.querySelectorAll('canvas[id*="matplotlib"]:not([id*="rubberband"])')
      
      canvases.forEach(canvas => {
        // Only move if it's not already in our container
        if (!container.contains(canvas)) {
          // Remove any existing canvases in container first
          const existingCanvases = container.querySelectorAll('canvas')
          existingCanvases.forEach(existing => existing.remove())
          
          // Move the new canvas to our container
          container.appendChild(canvas)
          
          // Add a download button for the plot
          const downloadBtn = document.createElement('button')
          downloadBtn.innerHTML = '📊 Save Plot'
          downloadBtn.className = 'mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded cursor-pointer'
          downloadBtn.onclick = () => {
            // Convert canvas to blob and download
            const canvas = container.querySelector('canvas') as HTMLCanvasElement
            if (canvas) {
              canvas.toBlob(blob => {
                if (blob) {
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = 'matplotlib_plot.png'
                  a.click()
                  URL.revokeObjectURL(url)
                }
              })
            }
          }
          container.appendChild(downloadBtn)
        }
      })
    }, 500) // Give matplotlib time to create the canvas
  }

  // Function to generate or edit code using AI
  const generateOrEditCode = async (message: string, existingCode?: string): Promise<{code: string, requiredPackages: string[]} | null> => {
    if (!apiUrl) return null
    
    try {
      // Check if user specifically requests plotting libraries
      const requestsPlotting = /matplotlib|seaborn|plotly|plot.*with|chart.*with|graph.*with/i.test(message)
      
      // Format the description for the API
      let description = message
      if (existingCode) {
        description = `${message}\n\nExisting code to modify:\n${existingCode}`
      }
      
      // Only add anti-plotting instruction if plotting is NOT specifically requested
      if (!requestsPlotting) {
        if (existingCode) {
          description += `\n\nIMPORTANT: Do not use matplotlib, seaborn, plotly or any plotting libraries. Focus on data processing, calculations, and text output only.`
        } else {
          description += `\n\nIMPORTANT: Do not use matplotlib, seaborn, plotly or any plotting libraries. Focus on data processing, calculations, and text output only. Use print() statements to show results.`
        }
      }
      
      const response = await fetch(`${apiUrl}/generate-code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description: description,
          language: 'python'
        })
      })
      
      console.log('API response status:', response.status) // Debug log
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        console.error('API error:', errorData) // Debug log
        throw new Error(errorData.detail || `HTTP ${response.status}: Code generation failed`)
      }

      const data = await response.json()
      console.log('API response data:', data) // Debug log      // Analyze the generated code for required packages
      const requiredPackages = analyzeCodePackages(data.code || '')
      
      return {
        code: data.code || '',
        requiredPackages
      }
    } catch (error) {
      console.error('Code generation error:', error)
      return null
    }
  }

  // Function to create a new code session
  const createCodeSession = (code: string, description: string, requiredPackages: string[]): string => {
    const sessionId = Date.now().toString()
    const newSession: CodeSession = {
      id: sessionId,
      code,
      description,
      requiredPackages,
      result: null,
      timestamp: Date.now(),
      versions: []
    }
    
    console.log('Creating new code session:', sessionId, 'with description:', description) // Debug log
    console.log('Session code:', code) // Debug log
    
    setCodeSessions(prev => [...prev, newSession])
    setActiveCodeId(sessionId)
    setEditingCode(code)
    
    console.log('Active code ID set to:', sessionId) // Debug log
    
    return sessionId
  }

  // Function to clear all code sessions
  const clearCodeSessions = () => {
    setCodeSessions([])
    setActiveCodeId(null)
    setEditingCode('')
    setCodeSessionActive(false)
    setCopiedCode(null)
    
    // Reset Python environment
    resetPyodideCompletely()
  }

  // Function to execute the edited code
  const executeEditedCode = async () => {
    if (!activeCodeId || !pyodideReady || executing) return
    
    const session = codeSessions.find(s => s.id === activeCodeId)
    if (!session) return
    
    setExecuting(true)
    setExecutionProgress(0)
    
    // Simple progress simulation
    const progressInterval = setInterval(() => {
      setExecutionProgress(prev => Math.min(prev + 25, 100))
    }, 250)
    
    try {
      const result = await executeCode(editingCode, session.requiredPackages)
      
      // Update session with result
      setCodeSessions(prev => prev.map(s => 
        s.id === activeCodeId ? { ...s, result, code: editingCode } : s
      ))
      
      // If the code contains plotting, move matplotlib canvases to our container
      if (containsPlottingCode(editingCode)) {
        moveMatplotlibCanvases()
      }
      
    } catch (error) {
      console.error('Execution error:', error)
    } finally {
      clearInterval(progressInterval)
      setExecuting(false)
      setExecutionProgress(0)
    }
  }

  // Set API URL after component mounts
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Always use Next.js API routes for consistency between dev and prod
      setApiUrl('/api')
    }
  }, [])

  // Truncate text to fit within character limit
  const truncateText = (text: string, maxLength: number): string => {
    if (text.length <= maxLength) return text
    return text.substring(0, maxLength - 3) + '...'
  }

  // AI-powered code action analysis
  const analyzeCodeAction = async (userMessage: string, hasActiveCode: boolean, codeContext?: string): Promise<CodeActionAnalysis> => {
    if (!apiUrl) {
      // Fallback to simple keyword matching if API not available
      return analyzeCodeActionFallback(userMessage, hasActiveCode)
    }
    
    try {
      const response = await fetch(`${apiUrl}/analyze-code-action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_message: userMessage,
          has_active_code: hasActiveCode,
          code_context: codeContext || null,
          conversation_context: messages.slice(-3).map(msg => ({
            role: msg.role,
            content: msg.content.substring(0, 200)
          }))
        })
      })
      
      if (!response.ok) {
        throw new Error('Analysis service unavailable')
      }
      
      const data = await response.json()
      
      return {
        action: data.action || 'none',
        confidence: data.confidence || 0.5,
        context: data.context || {}
      }
      
    } catch (error) {
      console.error('Code action analysis failed:', error)
      return analyzeCodeActionFallback(userMessage, hasActiveCode)
    }
  }

  // Fallback analysis using enhanced keyword detection
  const analyzeCodeActionFallback = (userMessage: string, hasActiveCode: boolean): CodeActionAnalysis => {
    const lowerText = userMessage.toLowerCase()
    
    // If PDF context is active, be VERY strict about code generation
    // Only generate code for explicit programming requests
    const hasPdfContext = !!pdfContext
    
    // Version/revert detection
    const versionKeywords = [
      'go back', 'previous version', 'revert', 'undo', 'roll back', 'restore',
      'previous', 'before', 'earlier', 'original', 'old version', 'last version'
    ]
    
    const executionKeywords = [
      'execute', 'run', 'run code', 'execute code', 'run this', 'execute this',
      'can you execute', 'can you run', 'code execute', 'run it', 'execute it'
    ]
    
    const editKeywords = [
      'edit', 'change', 'modify', 'update', 'alter', 'fix', 'replace',
      'make it', 'turn it into', 'convert to', 'switch to'
    ]
    
    // Much more strict code generation keywords - require explicit programming language
    const strictCodeKeywords = [
      'write python code', 'create python code', 'generate python code', 'python code for',
      'write javascript', 'create javascript', 'write a function', 'create a function',
      'code to calculate', 'python script', 'javascript function', 'write a program',
      'create a program', 'python program', 'coding challenge', 'programming challenge'
    ]
    
    // Simple code keywords - only use when NO PDF context
    const simpleCodeKeywords = hasPdfContext ? [] : [
      'print hello world', 'hello world code', 'fibonacci code', 'factorial code',
      'prime number code', 'calculate fibonacci', 'write fibonacci'
    ]
    
    const plottingKeywords = [
      'bar chart', 'line chart', 'pie chart', 'histogram', 'scatter plot',
      'plot', 'graph', 'chart', 'visualize', 'visualization', 'matplotlib',
      'draw a chart', 'create a chart', 'show a graph', 'make a plot'
    ]
    
    // Much more explicit code generation requests
    const explicitCodeKeywords = [
      'write code', 'create code', 'generate code', 'code example',
      'programming example', 'implement', 'algorithm implementation'
    ]
    
    // Check for version requests
    if (hasActiveCode && versionKeywords.some(kw => lowerText.includes(kw))) {
      return {
        action: 'revert',
        confidence: 0.9,
        context: {
          versionRequest: {
            type: 'previous',
            steps: 1
          }
        }
      }
    }
    
    // Check for execution requests
    if (hasActiveCode && executionKeywords.some(kw => lowerText.includes(kw))) {
      return {
        action: 'execute',
        confidence: 0.85,
        context: { needsExecution: true }
      }
    }
    
    // Check for edit requests (only if not asking for new code)
    if (hasActiveCode && editKeywords.some(kw => lowerText.includes(kw))) {
      return {
        action: 'edit',
        confidence: 0.8,
        context: { hasCodeContext: true }
      }
    }
    
    // STRICT code generation - only for explicit programming requests
    if (strictCodeKeywords.some(kw => lowerText.includes(kw))) {
      return {
        action: 'generate',
        confidence: 0.9,
        context: { isPlottingRelated: plottingKeywords.some(kw => lowerText.includes(kw)) }
      }
    }
    
    // Explicit code requests (but lower confidence when PDF is active)
    if (explicitCodeKeywords.some(kw => lowerText.includes(kw))) {
      return {
        action: 'generate',
        confidence: hasPdfContext ? 0.3 : 0.7, // Much lower confidence with PDF
        context: { isPlottingRelated: plottingKeywords.some(kw => lowerText.includes(kw)) }
      }
    }
    
    // Plotting requests (only if explicit)
    if (plottingKeywords.some(kw => lowerText.includes(kw)) && !hasPdfContext) {
      return {
        action: 'generate',
        confidence: 0.6,
        context: { isPlottingRelated: true }
      }
    }
    
    // Simple code requests (only when NO PDF context)
    if (simpleCodeKeywords.some(kw => lowerText.includes(kw))) {
      return {
        action: 'generate',
        confidence: 0.8,
        context: { isPlottingRelated: false }
      }
    }
    
    // Default to question/chat when PDF context is available or no clear code intent
    return {
      action: 'question',
      confidence: hasPdfContext ? 0.9 : 0.6, // High confidence for questions when PDF is active
      context: {}
    }
  }

  // Smart context creation for code questions
  const createSmartContext = (userMessage: string, codeId?: string): string => {
    const MAX_MESSAGE_LENGTH = 450
    let baseMessage = userMessage
    
    if (!codeId) return baseMessage
    
    const session = codeSessions.find(s => s.id === codeId)
    if (!session) return baseMessage
    
    const availableSpace = MAX_MESSAGE_LENGTH - baseMessage.length - 50
    if (availableSpace <= 0) return baseMessage
    
    let context = ''
    
    if (session.result && !session.result.success) {
      const errorInfo = ` [Error: ${session.result.error?.substring(0, 100)}...]`
      if (context.length + errorInfo.length < availableSpace) {
        context += errorInfo
      }
    }
    
    const remainingSpace = availableSpace - context.length
    if (remainingSpace > 50) {
      const codeSnippet = ` [Code: ${session.code.substring(0, remainingSpace - 20)}...]`
      context += codeSnippet
    }
    
    return baseMessage + context
  }

  // Analyze code to determine required packages (now including plotting libraries)
  const analyzeCodePackages = (code: string): string[] => {
    const packages = []
    const lines = code.toLowerCase()
    
    if (lines.includes('import numpy') || lines.includes('from numpy')) packages.push('numpy')
    if (lines.includes('import pandas') || lines.includes('from pandas')) packages.push('pandas')
    if (lines.includes('import scipy') || lines.includes('from scipy')) packages.push('scipy')
    if (lines.includes('import sklearn') || lines.includes('from sklearn')) packages.push('scikit-learn')
    if (lines.includes('import requests') || lines.includes('from requests')) packages.push('requests')
    if (lines.includes('import matplotlib') || lines.includes('from matplotlib')) packages.push('matplotlib')
    if (lines.includes('import seaborn') || lines.includes('from seaborn')) packages.push('seaborn')
    if (lines.includes('import plotly') || lines.includes('from plotly')) packages.push('plotly')
    
    return [...new Set(packages)]
  }

  // Check if code contains plotting/visualization libraries
  const containsPlottingCode = (code: string): boolean => {
    const plottingKeywords = [
      'import matplotlib', 'from matplotlib', 'import seaborn', 'from seaborn',
      'import plotly', 'from plotly', 'plt.', '.plot(', '.show(', '.savefig(',
      'seaborn.', 'sns.', 'plotly.'
    ]
    
    const lowerCode = code.toLowerCase()
    return plottingKeywords.some(keyword => lowerCode.includes(keyword))
  }

  // PDF Upload functionality
  const handlePdfUpload = async (file: File): Promise<void> => {
    if (!apiUrl) {
      setUploadError('API not available')
      return
    }

    setUploading(true)
    setUploadError('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const data: UploadResponse = await response.json()
      
      // Create PDF context
      const newPdfContext: PDFContext = {
        sessionId: data.session_id,
        filename: data.filename,
        summary: data.result,
        isActive: true,
        uploadedAt: Date.now()
      }

      setPdfContext(newPdfContext)

      // Add a message to the chat showing the PDF was uploaded
      const uploadMessage: ChatMessage = {
        role: 'assistant',
        content: `📄 **PDF Uploaded Successfully**\n\n**File:** ${data.filename}\n\n**Summary:**\n${data.result}\n\nYou can now ask questions about this document!`
      }

      setMessages(prev => [...prev, uploadMessage])

    } catch (error: any) {
      console.error('PDF Upload error:', error)
      setUploadError(error.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  // Clear PDF context
  const clearPdfContext = () => {
    setPdfContext(null)
    setUploadError('')
  }

  // Add version to code session
  const addVersionToSession = (sessionId: string, code: string, description: string, requiredPackages: string[]): void => {
    setCodeSessions(prev => prev.map(session => 
      session.id === sessionId 
        ? { 
            ...session, 
            versions: [...session.versions, {
              code: session.code,
              timestamp: session.timestamp,
              description: session.description,
              requiredPackages: session.requiredPackages
            }],
            code,
            description,
            requiredPackages,
            result: null,
            timestamp: Date.now()
          }
        : session
    ))
    setEditingCode(code)
  }

  // Revert to previous version
  const revertToPreviousVersion = (sessionId: string, steps: number = 1): boolean => {
    const session = codeSessions.find(s => s.id === sessionId)
    if (!session || session.versions.length === 0) return false
    
    const targetIndex = Math.max(0, session.versions.length - steps)
    const targetVersion = session.versions[targetIndex]
    
    if (!targetVersion) return false
    
    // Remove the versions we're reverting from
    const newVersions = session.versions.slice(0, targetIndex)
    
    setCodeSessions(prev => prev.map(s => 
      s.id === sessionId 
        ? { 
            ...s,
            code: targetVersion.code,
            description: targetVersion.description,
            requiredPackages: targetVersion.requiredPackages,
            result: null,
            timestamp: Date.now(),
            versions: newVersions
          }
        : s
    ))
    
    setEditingCode(targetVersion.code)
    return true
  }

  // Initialize Pyodide
  const initializePyodide = async () => {
    if (pyodideLoading) return
    
    setPyodideLoading(true)
    try {
      if (!window.pyodide) {
        const script = document.createElement('script')
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js'
        script.onload = async () => {
          try {
            window.pyodide = await window.loadPyodide()
            await window.pyodide.loadPackage(['numpy'])
            setLoadedPackages(new Set(['numpy']))
            setPyodideReady(true)
            console.log('Pyodide initialized successfully')
          } catch (error) {
            console.error('Failed to initialize Pyodide:', error)
            setPyodideReady(false)
          }
        }
        script.onerror = () => {
          console.error('Failed to load Pyodide script')
          setPyodideReady(false)
          setPyodideLoading(false)
        }
        document.head.appendChild(script)
      } else {
        setPyodideReady(true)
      }
    } catch (error) {
      console.error('Error loading Pyodide:', error)
      setPyodideReady(false)
    } finally {
      setPyodideLoading(false)
    }
  }

  // Load packages on demand (now including plotting libraries)
  const loadPackageIfNeeded = async (packages: string[]): Promise<void> => {
    const packagesToLoad = packages.filter(pkg => !loadedPackages.has(pkg))
    
    if (packagesToLoad.length === 0) return
    
    try {
      for (const pkg of packagesToLoad) {
        await window.pyodide.loadPackage([pkg])
        setLoadedPackages(prev => new Set([...prev, pkg]))
      }
      console.log('Loaded packages:', packagesToLoad)
    } catch (error) {
      console.error('Failed to load packages:', packagesToLoad, error)
    }
  }

  // Enhanced code execution with plotting support
  const executeCode = async (code: string, requiredPackages: string[] = []): Promise<ExecutionResult> => {
    if (!pyodideReady) {
      return {
        success: false,
        error: "Python environment not ready yet. Please wait a moment.",
        executionTime: 0
      }
    }

    // Check for dangerous patterns before execution (but allow plotting)
    const warnings = detectPotentialInfiniteLoops(code)
    if (warnings.length > 0) {
      return {
        success: false,
        error: `❌ Code execution blocked for safety:\n\n${warnings.join('\n')}\n\n🔒 This code contains patterns that could freeze your browser. Please modify the code to:\n• Add break conditions to loops\n• Reduce large ranges\n• Avoid nested infinite loops\n\nOnce fixed, you can run the code safely.`,
        executionTime: 0
      }
    }
    
    const startTime = performance.now()
    
    try {
      // Load all required packages (including plotting libraries)
      await loadPackageIfNeeded(requiredPackages)
      
      // Set up clean execution environment
      await window.pyodide.runPython(`
import sys
from io import StringIO

# Backup stdout/stderr
_old_stdout = sys.stdout
_old_stderr = sys.stderr
_stdout_buffer = StringIO()
_stderr_buffer = StringIO()
sys.stdout = _stdout_buffer
sys.stderr = _stderr_buffer
      `)
      
      // Execute the code directly
      await window.pyodide.runPython(code)
      
      // Get output
      const stdout = await window.pyodide.runPython('_stdout_buffer.getvalue()')
      const stderr = await window.pyodide.runPython('_stderr_buffer.getvalue()')
      
      // Restore stdout/stderr
      await window.pyodide.runPython(`
sys.stdout = _old_stdout
sys.stderr = _old_stderr
      `)
      
      const executionTime = performance.now() - startTime
      
      let finalOutput = ''
      if (stdout) finalOutput += stdout
      if (stderr) finalOutput += (finalOutput ? '\n' : '') + stderr
      
      // Check if matplotlib plot was created and saved
      if (containsPlottingCode(code)) {
        try {
          // Try to get the plot as base64 if it was saved
          const plotExists = await window.pyodide.runPython(`
import os
'plot.png' in os.listdir() if os.path.exists('.') else False
          `)
          
          if (plotExists) {
            finalOutput += '\n\n📊 Plot saved successfully as plot.png'
            // You could potentially convert the plot to base64 and display it here
          }
        } catch (plotError) {
          console.log('Plot check failed:', plotError)
        }
      }
      
      if (!finalOutput.trim()) {
        finalOutput = 'Code executed successfully (no output)'
      }
      
      return {
        success: true,
        output: finalOutput,
        executionTime
      }
      
    } catch (error: any) {
      try {
        await window.pyodide.runPython(`
sys.stdout = _old_stdout
sys.stderr = _old_stderr
        `)
      } catch {}
      
      return {
        success: false,
        error: error.message || String(error),
        executionTime: performance.now() - startTime
      }
    }
  }

  // Function to instrument code with interrupt checks
  const addInterruptChecks = (code: string): string => {
    // This function is no longer used but kept for potential future use
    return code
  }

  // Nuclear option: completely reset Pyodide
  const resetPyodideCompletely = async () => {
    try {
      setExecuting(false)
      setExecutionProgress(0)
      setPyodideReady(false)
      setPyodideLoading(false)
      setLoadedPackages(new Set())
      
      // Clear the global pyodide object
      if (window.pyodide) {
        delete window.pyodide
      }
      
      // Remove pyodide script tag if it exists
      const existingScript = document.querySelector('script[src*="pyodide"]')
      if (existingScript) {
        existingScript.remove()
      }
      
      // Force garbage collection
      if (window.gc) {
        window.gc()
      }
      
      console.log('Pyodide completely reset - reinitialize to continue')
      
    } catch (error) {
      console.error('Failed to reset Pyodide:', error)
    }
  }

  // Handle submit
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
      // Get code context for potential use later
      const codeContext = activeCodeId ? codeSessions.find(s => s.id === activeCodeId)?.code : undefined
      
      // When PDF context is active, only analyze for explicit code requests
      // Skip analysis for general questions to avoid code generation
      const shouldAnalyzeForCode = !pdfContext || 
        /write.*code|create.*code|generate.*code|python|javascript|function|program|script/.test(userMessage.toLowerCase())
      
      let analysis: CodeActionAnalysis
      
      if (shouldAnalyzeForCode) {
        // Analyze user intent for code-related actions
        analysis = await analyzeCodeAction(userMessage, !!activeCodeId, codeContext)
      } else {
        // For PDF context questions, force it to be a question
        analysis = {
          action: 'question',
          confidence: 0.95,
          context: {}
        }
      }
      
      console.log('🔍 Analysis Debug:')
      console.log('  - Action:', analysis.action)
      console.log('  - Confidence:', analysis.confidence)
      console.log('  - Has active code:', !!activeCodeId)
      console.log('  - Has PDF context:', !!pdfContext)
      console.log('  - PDF filename:', pdfContext?.filename || 'None')
      console.log('  - Should analyze for code:', shouldAnalyzeForCode)
      console.log('  - User message:', userMessage)
      console.log('  - Analysis:', analysis)

      // Handle different actions based on AI analysis
      switch (analysis.action) {
        case 'generate':
          // Clear search info when doing code generation
          setLastSearchInfo(null)
          
          // Initialize Python if needed for code generation
          if (!pyodideReady && !pyodideLoading) {
            setCodeSessionActive(true)
            await initializePyodide()
          }
          
          const smartMessage = createSmartContext(userMessage, activeCodeId || undefined)
          const codeResult = await generateOrEditCode(smartMessage)
          
          console.log('Code generation result:', codeResult) // Debug log
          
          if (codeResult) {
            const sessionId = createCodeSession(
              codeResult.code, 
              userMessage.length > 50 ? userMessage.substring(0, 50) + '...' : userMessage,
              codeResult.requiredPackages
            )
            
            console.log('Created session:', sessionId, 'with code:', codeResult.code) // Debug log
            console.log('Current active code ID before:', activeCodeId) // Debug log
            console.log('Current code sessions before:', codeSessions.length) // Debug log
            
            setCodeSessionActive(true)
            
            // Force a slight delay to ensure state updates
            setTimeout(() => {
              console.log('After state update - Active code ID:', activeCodeId) // Debug log
              console.log('After state update - Code sessions:', codeSessions.length) // Debug log
            }, 100)
            
            // Add assistant response with code reference
            const responseMessage = `I've generated Python code for you! Here's what I created:\n\n\`\`\`python\n${codeResult.code}\n\`\`\`\n\nThe code is now available in the editor where you can run it, edit it, or ask for modifications. ${containsPlottingCode(codeResult.code) ? '\n\n⚠️ Note: This code contains plotting functions which cannot be executed in the browser, but you can copy it for local use.' : ''}`
            
            console.log('Response message created with code:', codeResult.code) // Debug log
            console.log('Full response message:', responseMessage) // Debug log
            
            setMessages([...newMessages, { 
              role: 'assistant', 
              content: responseMessage,
              hasCode: true,
              codeId: sessionId
            }])
            
            setLoading(false)
            return
          } else {
            // If code generation failed, fall back to regular chat with an explanation
            const fallbackMessage = "I couldn't generate code right now due to a backend service issue. However, I can still help you with coding questions, explanations, and guidance! What would you like to know about programming?"
            
            setMessages([...newMessages, { 
              role: 'assistant', 
              content: fallbackMessage
            }])
            
            setLoading(false)
            return
          }
          break

        case 'edit':
          // Clear search info when editing code
          setLastSearchInfo(null)
          
          if (activeCodeId && codeContext) {
            const editResult = await generateOrEditCode(userMessage, codeContext)
            
            if (editResult) {
              // Add version to session before editing
              addVersionToSession(
                activeCodeId,
                editResult.code,
                userMessage.length > 50 ? userMessage.substring(0, 50) + '...' : userMessage,
                editResult.requiredPackages
              )
              
              const responseMessage = `I've updated your code based on your request:\n\n\`\`\`python\n${editResult.code}\n\`\`\`\n\nThe changes are now in the editor. You can run it or ask for further modifications!`
              
              setMessages([...newMessages, { 
                role: 'assistant', 
                content: responseMessage,
                hasCode: true,
                codeId: activeCodeId
              }])
              
              setLoading(false)
              return
            } else {
              // If edit failed, provide helpful message
              const editFailedMessage = "I couldn't edit the code right now due to a backend service issue. You can manually edit the code in the editor above, or I can help explain how to make the changes you want!"
              
              setMessages([...newMessages, { 
                role: 'assistant', 
                content: editFailedMessage
              }])
              
              setLoading(false)
              return
            }
          }
          break

        case 'execute':
          // Clear search info when executing code
          setLastSearchInfo(null)
          
          if (activeCodeId) {
            if (!pyodideReady) {
              setMessages([...newMessages, { 
                role: 'assistant', 
                content: "Python environment is still loading. Please wait a moment and try again."
              }])
              setLoading(false)
              return
            }
            
            const session = codeSessions.find(s => s.id === activeCodeId)
            if (session) {
              setExecuting(true)
              const result = await executeCode(session.code, session.requiredPackages)
              
              // Update session with execution result
              setCodeSessions(prev => prev.map(s => 
                s.id === activeCodeId ? { ...s, result } : s
              ))
              
              const responseMessage = result.success 
                ? `✅ Code executed successfully!\n\nOutput:\n\`\`\`\n${result.output}\n\`\`\`\n\nExecution time: ${result.executionTime?.toFixed(2)}ms`
                : `❌ Code execution failed:\n\n\`\`\`\n${result.error}\n\`\`\`\n\nYou can edit the code and try again.`
            
              setMessages([...newMessages, { 
                role: 'assistant', 
                content: responseMessage,
                hasCode: true,
                codeId: activeCodeId
              }])
              
              setExecuting(false)
              setLoading(false)
              return
            }
          }
          break

        case 'revert':
          // Clear search info when reverting code
          setLastSearchInfo(null)
          
          if (activeCodeId && analysis.context?.versionRequest) {
            const steps = analysis.context.versionRequest.steps || 1
            const success = revertToPreviousVersion(activeCodeId, steps)
            
            if (success) {
              const session = codeSessions.find(s => s.id === activeCodeId)
              const responseMessage = `✅ Reverted to previous version!\n\n\`\`\`python\n${session?.code}\n\`\`\`\n\nThe previous code is now active in the editor.`
              
              setMessages([...newMessages, { 
                role: 'assistant', 
                content: responseMessage,
                hasCode: true,
                codeId: activeCodeId
              }])
              
              setLoading(false)
              return
            } else {
              setMessages([...newMessages, { 
                role: 'assistant', 
                content: "❌ No previous versions available to revert to."
              }])
              
              setLoading(false)
              return
            }
          }
          break

        default:
          // For questions and non-code requests, use regular chat
          break
      }

      // Fall back to regular chat for questions and non-code requests
      const response = await fetch('/api/chat-search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          messages: newMessages,
          search_when_needed: true,
          num_search_results: 5,
          session_id: pdfContext?.sessionId || undefined
        }),
      })

      if (!response.ok) {
        throw new Error('Chat failed')
      }

      const data: ChatWithSearchResponse = await response.json()
      
      // Add assistant response to conversation
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: pdfContext 
          ? `📄 **Based on ${pdfContext.filename}:**\n\n${data.ai_response}`
          : data.ai_response
      }
      setMessages([...newMessages, assistantMessage])
      
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
      {/* Style matplotlib canvases to appear in the output panel */}
      <style>{`
        /* Target matplotlib canvas elements */
        canvas[id*="matplotlib"] {
          position: relative !important;
          left: auto !important;
          top: auto !important;
          z-index: 1 !important;
          max-width: 100% !important;
          height: auto !important;
          width: auto !important;
          border-radius: 8px;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          background: white;
          margin: 10px 0;
        }
        
        /* Hide the rubberband overlay since we don't need interactivity */
        canvas[id*="rubberband"] {
          display: none !important;
        }
        
        /* Hide matplotlib toolbar and figure title */
        div[id*="matplotlib"] div[id*="top"] {
          display: none !important;
        }
        
        /* Hide the entire toolbar container */
        div[id*="matplotlib"] > div:last-child {
          display: none !important;
        }
        
        /* Hide matplotlib toolbar buttons */
        .matplotlib-toolbar-button {
          display: none !important;
        }
        
        /* Hide matplotlib message div */
        div[id*="message"] {
          display: none !important;
        }
        
        /* Ensure matplotlib elements appear in the output container */
        .matplotlib-container canvas[id*="matplotlib"] {
          display: block;
          margin: 0 auto;
        }
        
        /* Style the matplotlib container itself */
        div[id*="matplotlib"] {
          background: transparent !important;
          border: none !important;
          margin: 10px 0 !important;
        }
        
        /* Custom scrollbar styles for minimalistic look */
        .scrollbar-thin {
          scrollbar-width: thin;
        }
        
        .scrollbar-thin::-webkit-scrollbar {
          width: 6px;
        }
        
        .scrollbar-thin::-webkit-scrollbar-track {
          background: transparent;
        }
        
        .scrollbar-thin::-webkit-scrollbar-thumb {
          background-color: rgba(255, 255, 255, 0.2);
          border-radius: 3px;
          transition: background-color 0.2s ease;
        }
        
        .scrollbar-thin::-webkit-scrollbar-thumb:hover {
          background-color: rgba(255, 255, 255, 0.3);
        }
        
        /* Hide scrollbar when not needed */
        .scrollbar-thin::-webkit-scrollbar {
          width: 0px;
          background: transparent;
        }
        
        .scrollbar-thin:hover::-webkit-scrollbar {
          width: 6px;
        }
        
        /* For Firefox */
        .scrollbar-thin {
          scrollbar-width: none;
        }
        
        .scrollbar-thin:hover {
          scrollbar-width: thin;
          scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
        }
      `}</style>
      
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <MessageCircle className="w-8 h-8 text-purple-400" />
            <h1 className="text-4xl font-bold text-white">kh.AI Search & Code</h1>
          </div>
          <p className="text-gray-300 text-lg">
            Intelligent Assistant. Chat, Search, and Code in Python.
          </p>
          
          {/* PDF Upload Section */}
          <div className="mt-6 flex justify-center">
            <div className="flex items-center gap-4 bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl p-4 max-w-2xl">
              {!pdfContext ? (
                <>
                  <div className="flex items-center gap-2">
                    <FileText className="w-5 h-5 text-purple-400" />
                    <span className="text-white font-medium">Upload PDF</span>
                  </div>
                  <label className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg cursor-pointer transition-colors">
                    <Upload className="w-4 h-4 text-white" />
                    <span className="text-white">Choose File</span>
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(e) => {
                        const file = e.target.files?.[0]
                        if (file) {
                          handlePdfUpload(file)
                          e.target.value = ''
                        }
                      }}
                      className="hidden"
                      disabled={uploading}
                    />
                  </label>
                  {uploading && (
                    <div className="flex items-center gap-2 text-blue-400">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">Uploading...</span>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <FileText className="w-5 h-5 text-green-400" />
                    <div className="text-left">
                      <div className="text-white font-medium text-sm">{pdfContext.filename}</div>
                      <div className="text-green-400 text-xs">PDF Active</div>
                    </div>
                  </div>
                  <button
                    onClick={clearPdfContext}
                    className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-white text-sm transition-colors flex items-center gap-1"
                  >
                    <X className="w-3 h-3" />
                    Remove
                  </button>
                </>
              )}
            </div>
          </div>
          
          {uploadError && (
            <div className="mt-4 max-w-2xl mx-auto">
              <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-3 text-red-200 text-sm flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{uploadError}</span>
              </div>
            </div>
          )}
          
          {/* Enhanced Python Status */}
          {codeSessionActive && (
            <div className="mt-4 flex justify-center">
              {pyodideLoading && (
                <div className="text-blue-400 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading Python environment...
                </div>
              )}
              
              {pyodideReady && (
                <div className="text-green-400 flex items-center gap-2">
                  <Terminal className="w-4 h-4" />
                  Python ready
                </div>
              )}
              
              {!pyodideReady && !pyodideLoading && codeSessionActive && (
                <div className="text-yellow-400 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  Python initializing...
                </div>
              )}
            </div>
          )}
        </div>

        <div className={`grid gap-6 ${codeSessionActive ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
          {/* Chat Section */}
          <div className={`bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl p-6 ${codeSessionActive ? 'h-[calc(100vh-12rem)]' : 'h-auto'} flex flex-col`}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
          <MessageCircle className="w-5 h-5 text-purple-400" />
          <h2 className="text-xl font-semibold text-white">Chat</h2>
              </div>
              <div className="flex items-center gap-3">
                {pdfContext && (
                  <div className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded-full flex items-center gap-1">
                    <FileText className="w-3 h-3" />
                    PDF Active
                  </div>
                )}
                {codeSessionActive && (
                  <div className="text-xs text-gray-400">
                    AI Code Assistant Active →
                  </div>
                )}
              </div>
            </div>
            
            {/* Chat History */}
            <div 
              ref={chatContainerRef}
              className={`${codeSessionActive ? 'flex-1' : 'h-[32rem]'} overflow-y-auto mb-4 space-y-2 scroll-smooth bg-gradient-to-b from-transparent via-black/5 to-black/10 rounded-lg p-3 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/20 hover:scrollbar-thumb-white/30`}
              style={{ scrollBehavior: 'smooth' }}
            >
              {messages.length === 0 ? (
                <div className="text-gray-400 px-6 py-20 text-center">
                  <div className="bg-white/5 rounded-full w-20 h-20 mx-auto mb-6 flex items-center justify-center">
                    <MessageCircle className="w-10 h-10 opacity-50" />
                  </div>
                  <h3 className="text-lg font-medium text-white mb-4">Start your AI conversation</h3>
                  <p className="text-gray-400 mb-6">Ask me anything, upload PDFs, or request code generation</p>
                  <div className="text-sm space-y-2 bg-white/5 rounded-lg p-4 max-w-sm mx-auto">
                    <div className="flex items-center gap-2 text-purple-300">
                      <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                      <span>Natural language: "Print hello world"</span>
                    </div>
                    <div className="flex items-center gap-2 text-blue-300">
                      <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                      <span>Execution: "Run this code"</span>
                    </div>
                    <div className="flex items-center gap-2 text-green-300">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>Code editing: "Change the function"</span>
                    </div>
                    <div className="flex items-center gap-2 text-orange-300">
                      <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
                      <span>PDF chat: Upload documents above</span>
                    </div>
                    <div className="flex items-center gap-2 text-purple-400 font-medium mt-3">
                      <span>🤖</span>
                      <span>Powered by intelligent context understanding</span>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  {messages.map((message, index) => (
                    <div
                      key={index}
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}
                    >
                      <div
                        className={`max-w-[85%] p-4 rounded-lg shadow-lg ${
                          message.role === 'user'
                            ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white'
                            : 'bg-white/10 text-gray-200 border border-white/20 backdrop-blur-sm'
                        }`}
                      >
                        <div className="whitespace-pre-wrap">
                          {message.content.split(/(\`\`\`[\s\S]*?\`\`\`|\*\*[^*]+\*\*)/g).map((part, partIndex) => {
                            // Handle code blocks
                            if (part.startsWith('```') && part.endsWith('```')) {
                              const codeContent = part.slice(3, -3)
                              const lines = codeContent.split('\n')
                              const language = lines[0]
                              const code = lines.slice(1).join('\n')
                              
                              return (
                                <div key={partIndex} className="my-3">
                                  <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
                                    {language && (
                                      <div className="bg-gray-800 px-3 py-1 text-xs text-gray-300 border-b border-gray-700">
                                        {language}
                                      </div>
                                    )}
                                    <pre className="p-3 text-green-400 font-mono text-sm overflow-x-auto">
                                      <code>{code}</code>
                                    </pre>
                                  </div>
                                </div>
                              )
                            }
                            // Handle bold text
                            else if (part.startsWith('**') && part.endsWith('**')) {
                              return (
                                <strong key={partIndex} className="font-semibold text-white">
                                  {part.slice(2, -2)}
                                </strong>
                              )
                            }
                            // Regular text
                            return <span key={partIndex}>{part}</span>
                          })}
                        </div>
                        
                        {message.hasCode && message.codeId && (
                          <div className="mt-3 p-3 bg-white/10 rounded-lg border border-white/20">
                            <div className="flex items-center gap-2 text-sm">
                              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                              <Code className="w-4 h-4 text-green-400" />
                              <span className="text-green-400 font-medium">Code ready in editor</span>
                              <div className="ml-auto">
                                <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {loading && (
                    <div className="flex justify-start mb-4">
                      <div className="bg-white/10 border border-white/20 rounded-lg p-4 backdrop-blur-sm shadow-lg">
                        <div className="flex items-center gap-3 text-gray-300">
                          <div className="relative">
                            <Loader2 className="w-5 h-5 animate-spin text-purple-400" />
                            <div className="absolute inset-0 w-5 h-5 border-2 border-purple-400/20 rounded-full"></div>
                          </div>
                          <div className="flex flex-col">
                            <span className="text-sm font-medium">AI is thinking...</span>
                            <span className="text-xs text-gray-400">Searching and analyzing when needed</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} style={{ height: '1px' }} />
                </>
              )}
            </div>

            {/* Search Status */}
            {lastSearchInfo && messages.length > 0 && (
              <div className="mb-4 p-4 bg-gradient-to-r from-white/5 to-white/10 border border-white/10 rounded-lg backdrop-blur-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {lastSearchInfo.performed ? (
                      <>
                        <div className="w-8 h-8 bg-green-500/20 rounded-full flex items-center justify-center">
                          <Search className="w-4 h-4 text-green-400" />
                        </div>
                        <div className="flex flex-col">
                          <span className="text-sm font-medium text-gray-200">Live search performed</span>
                          {lastSearchInfo.query && (
                            <span className="text-xs text-purple-300">"{lastSearchInfo.query}"</span>
                          )}
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="w-8 h-8 bg-blue-500/20 rounded-full flex items-center justify-center">
                          <MessageCircle className="w-4 h-4 text-blue-400" />
                        </div>
                        <div className="flex flex-col">
                          <span className="text-sm font-medium text-gray-200">Used existing knowledge</span>
                          <span className="text-xs text-gray-400">No search needed</span>
                        </div>
                      </>
                    )}
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-gray-400">Response time</div>
                    <div className="text-sm font-mono text-gray-300">
                      {lastSearchInfo.responseTime.toFixed(2)}s
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Input Form */}
            <div className="mt-auto">
              <form onSubmit={handleSubmit} className="flex gap-3">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={
                    pdfContext 
                      ? `Ask questions about ${pdfContext.filename} or anything else...`
                      : activeCodeId 
                        ? "Continue the conversation or request code changes..." 
                        : "Ask me anything, upload PDFs, or request code generation..."
                  }
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200 backdrop-blur-sm"
                  disabled={loading}
                  maxLength={450}
                />
                {input.length > 350 && (
                  <div className={`absolute -bottom-6 right-0 text-xs ${input.length > 430 ? 'text-red-400' : 'text-yellow-400'}`}>
                    {input.length}/450 characters
                  </div>
                )}
              </div>
              <button
                type="submit"
                disabled={loading || !input.trim() || input.length > 450}
                className="px-6 py-3 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 disabled:from-gray-600 disabled:to-gray-700 rounded-lg transition-all duration-200 transform hover:scale-105 disabled:scale-100 shadow-lg"
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
                  className="px-4 py-3 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-lg"
                  title="Clear conversation"
                >
                  <Trash2 className="w-5 h-5 text-white" />
                </button>
              )}
            </form>
            </div>

            {/* Error Message */}
            {error && (
              <div className="mt-4 p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-red-200 text-sm flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Code Panel */}
          {codeSessionActive && (
            <div className="space-y-6">
              {/* Code Editor */}
              <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Code className="w-5 h-5 text-green-400" />
                    <h2 className="text-xl font-semibold text-white">Smart Code Editor</h2>
                  </div>
                  <div className="flex items-center gap-2">
                    {/* Version indicator */}
                    {activeCodeId && (() => {
                      const session = codeSessions.find(s => s.id === activeCodeId)
                      return session && session.versions.length > 0 ? (
                        <div className="text-xs text-gray-400 flex items-center gap-1">
                          <RotateCcw className="w-3 h-3" />
                          Version {session.versions.length + 1}
                        </div>
                      ) : null
                    })()}
                    <button
                      onClick={clearCodeSessions}
                      className="bg-red-600 text-white px-3 py-1 rounded text-sm hover:bg-red-700 flex items-center gap-1"
                      title="Close code panel and clear Python session"
                    >
                      <X className="w-3 h-3" />
                      Close
                    </button>
                  </div>
                </div>

                {/* Code Sessions Tabs */}
                {codeSessions.length > 0 && (
                  <div className="mb-4">
                    <div className="flex gap-2 overflow-x-auto">
                      {codeSessions.map((session) => (
                        <button
                          key={session.id}
                          onClick={() => {
                            setActiveCodeId(session.id)
                            setEditingCode(session.code)
                          }}
                          className={`px-3 py-1 rounded text-sm whitespace-nowrap ${
                            activeCodeId === session.id
                              ? 'bg-purple-600 text-white'
                              : 'bg-white/10 text-gray-300 hover:bg-white/20'
                          }`}
                        >
                          {session.description.substring(0, 20)}...
                          {session.versions.length > 0 && (
                            <span className="ml-1 text-xs opacity-70">
                              v{session.versions.length + 1}
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Code Editor */}
                {activeCodeId ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between text-sm text-gray-400">
                      <div className="flex items-center gap-2">
                        <Edit3 className="w-4 h-4" />
                        <span>AI-powered code editor with version control</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => copyCodeToClipboard(editingCode)}
                          className="flex items-center gap-1 px-2 py-1 bg-white/10 rounded hover:bg-white/20"
                        >
                          {copiedCode === editingCode ? (
                            <Check className="w-3 h-3 text-green-400" />
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                          Copy
                        </button>
                      </div>
                    </div>
                    
                    <textarea
                      value={editingCode}
                      onChange={(e) => setEditingCode(e.target.value)}
                      className="w-full h-64 p-4 bg-gray-900 border border-gray-600 rounded text-green-400 font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-purple-500"
                      placeholder="Your code will appear here..."
                      style={{ fontFamily: 'Monaco, Consolas, "Lucida Console", monospace' }}
                    />

                    <div className="flex gap-2">
                      {executing ? (
                        // Show loading state during execution
                        <button
                          disabled
                          className="flex-1 py-3 rounded font-semibold flex items-center justify-center gap-2 bg-gray-600 text-white cursor-not-allowed"
                        >
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Executing...
                        </button>
                      ) : (
                        <button
                          onClick={async () => {
                            // Check for potential infinite loops before execution
                            const warnings = detectPotentialInfiniteLoops(editingCode)
                            
                            if (warnings.length > 0) {
                              const warningMessage = `⚠️ Potential infinite loop detected:\n${warnings.join('\n')}\n\nThis code is blocked for safety. Please fix the issues and try again.`
                              alert(warningMessage)
                              return
                            }
                            
                            await executeEditedCode()
                          }}
                          disabled={executing || !pyodideReady}
                          className="flex-1 py-3 rounded font-semibold flex items-center justify-center gap-2 bg-green-600 text-white hover:bg-green-700 disabled:bg-gray-600"
                        >
                          {pyodideReady ? (
                            <>
                              <Play className="w-4 h-4" />
                              Run Code {containsPlottingCode(editingCode) ? '(with Plotting)' : '(Safe Mode)'}
                            </>
                          ) : (
                            <>
                              <Loader2 className="w-4 h-4 animate-spin" />
                              Python Loading...
                            </>
                          )}
                        </button>
                      )}
                    </div>
                    
                    {/* Execution progress indicator */}
                    {executing && (
                      <div className="mt-2">
                        <div className="flex justify-between text-xs text-gray-400 mb-1">
                          <span>Executing code safely...</span>
                          <span>Processing</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <div 
                            className="h-2 rounded-full transition-all duration-300 bg-blue-500"
                            style={{ width: `${executionProgress}%` }}
                          />
                        </div>
                      </div>
                    )}
                    
                    {/* Enhanced execution guidelines */}
                    <div className="text-xs text-gray-400 bg-white/5 rounded p-3">
                      <strong className="text-white">AI Assistant Features:</strong>
                      <div className="mt-1 space-y-1">
                        <div>🤖 Natural language understanding</div>
                        <div>🔄 Automatic version control</div>
                        <div>✅ Contextual code editing</div>
                        <div>🔒 Safe execution (infinite loops blocked)</div>
                        <div>📊 Plotting libraries supported (matplotlib, seaborn, plotly)</div>
                        <div>💾 Auto-save plots as PNG files</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-gray-400 text-center py-16">
                    <Code className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Ask for code in natural language to start...</p>
                    <p className="text-sm mt-2">Try: "Print hello world" or "Calculate fibonacci"</p>
                  </div>
                )}
              </div>

              {/* Output Panel */}
              <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Terminal className="w-5 h-5 text-green-400" />
                  <h2 className="text-xl font-semibold text-white">Output & Visualization</h2>
                </div>
                
                {/* Matplotlib container for external canvases */}
                <div id="matplotlib-output-container" className="matplotlib-container mb-4">
                  {/* Matplotlib canvases will be moved here via JavaScript */}
                </div>
                
                <div className="bg-gray-900 text-green-400 p-4 rounded h-64 overflow-y-auto font-mono text-sm">
                  {activeCodeId && codeSessions.find(s => s.id === activeCodeId)?.result ? (
                    (() => {
                      const result = codeSessions.find(s => s.id === activeCodeId)?.result
                      return (
                        <div>
                          {result?.success ? (
                            <div>
                              <div className="text-green-400 mb-2">
                                ✅ Executed successfully in {result.executionTime?.toFixed(2)}ms
                              </div>
                              <pre className="whitespace-pre-wrap text-white">
                                {result.output}
                              </pre>
                            </div>
                          ) : (
                            <div>
                              <div className="text-red-400 mb-2">
                                ❌ Execution failed in {result?.executionTime?.toFixed(2)}ms
                              </div>
                              <pre className="whitespace-pre-wrap text-red-300">
                                {result?.error}
                              </pre>
                            </div>
                          )}
                        </div>
                      )
                    })()
                  ) : (
                    <div className="text-gray-500">
                      {activeCodeId 
                        ? pyodideReady 
                          ? "Click 'Run Code' to execute your code..." 
                          : "Waiting for Python to load..."
                        : "Terminal output and visualizations will appear here..."}
                    </div>
                  )}
                </div>

                {/* Code Sessions History with version info */}
                {codeSessions.length > 1 && (
                  <div className="mt-4">
                    <h3 className="font-semibold mb-2 text-white text-sm">Code Sessions</h3>
                    <div className="space-y-2 max-h-24 overflow-y-auto">
                      {codeSessions.slice().reverse().map((session) => (
                        <div 
                          key={session.id}
                          className={`text-sm p-2 rounded cursor-pointer transition-colors ${
                            activeCodeId === session.id
                              ? 'bg-purple-600/30 border border-purple-500'
                              : 'bg-white/5 hover:bg-white/10'
                          }`}
                          onClick={() => {
                            setActiveCodeId(session.id)
                            setEditingCode(session.code)
                          }}
                        >
                          <span className={session.result?.success ? 'text-green-400' : session.result ? 'text-red-400' : 'text-gray-400'}>
                            {session.result?.success ? '✅' : session.result ? '❌' : '⚪'}
                          </span>
                          <span className="ml-2 text-gray-400">
                            {new Date(session.timestamp).toLocaleTimeString()}
                          </span>
                          <span className="ml-2 text-gray-300">
                            {session.description.substring(0, 30)}...
                          </span>
                          {session.versions.length > 0 && (
                            <span className="ml-2 text-xs text-purple-300">
                              v{session.versions.length + 1}
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Sources section */}
        {lastSearchInfo?.performed && lastSearchInfo.sources.length > 0 && (
          <div className="mt-6 bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl p-6">
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