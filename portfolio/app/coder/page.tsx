"use client";

import { useState, useEffect, useRef } from 'react';

declare global {
  interface Window {
    pyodide: any;
    loadPyodide: () => Promise<any>;
  }
}

interface ExecutionResult {
  success: boolean;
  output?: string;
  error?: string;
  executionTime?: number;
}

export default function CodePlayground() {
  const [apiUrl, setApiUrl] = useState<string>("");
  const [pyodideReady, setPyodideReady] = useState(false);
  const [pyodideLoading, setPyodideLoading] = useState(false);
  
  // Code states
  const [code, setCode] = useState(`# Welcome to Python Code Playground!
# This runs Python directly in your browser using Pyodide

import math
import matplotlib.pyplot as plt
import numpy as np

# Example: Create a simple plot
x = np.linspace(0, 2 * math.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.plot(x, np.cos(x), 'r--', linewidth=2, label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine and Cosine Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Plot created successfully!")
print(f"Max sin value: {max(y):.3f}")
print(f"Min sin value: {min(y):.3f}")
`);
  
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [executing, setExecuting] = useState(false);
  
  // AI Generation states
  const [aiDescription, setAiDescription] = useState("");
  const [generatingCode, setGeneratingCode] = useState(false);
  
  // History
  const [codeHistory, setCodeHistory] = useState<Array<{code: string, result: ExecutionResult, timestamp: number}>>([]);
  
  const outputRef = useRef<HTMLDivElement>(null);

  // Set API URL after component mounts
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const port = window.location.hostname === 'localhost' ? '8000' : '8000';
      setApiUrl(`http://${window.location.hostname}:${port}`);
    }
  }, []);

  // Load Pyodide
  const initializePyodide = async () => {
    if (pyodideReady || pyodideLoading) return;
    
    setPyodideLoading(true);
    try {
      // Load Pyodide from CDN
      if (!window.pyodide) {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
        script.onload = async () => {
          try {
            window.pyodide = await window.loadPyodide();
        
            // Install common packages
            await window.pyodide.loadPackage(['numpy', 'matplotlib', 'pandas', 'scipy']);
            
            // Set up matplotlib backend for web
            await window.pyodide.runPython(`
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def show_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return f"data:image/png;base64,{img_base64}"

# Override plt.show() to return base64 image
original_show = plt.show
def custom_show(*args, **kwargs):
    return show_plot()
plt.show = custom_show
            `);
            
            setPyodideReady(true);
            console.log('Pyodide initialized successfully');
          } catch (error) {
            console.error('Failed to initialize Pyodide:', error);
          }
        };
        document.head.appendChild(script);
      } else {
        setPyodideReady(true);
      }
    } catch (error) {
      console.error('Error loading Pyodide:', error);
    } finally {
      setPyodideLoading(false);
    }
  };

  // Execute code in Pyodide
  const executeCode = async () => {
    if (!pyodideReady || executing) return;
    
    setExecuting(true);
    setExecutionResult(null);
    
    const startTime = performance.now();
    
    try {
      // Redirect stdout to capture print statements
      await window.pyodide.runPython(`
import sys
from io import StringIO
_stdout = sys.stdout
sys.stdout = StringIO()
_plot_data = None
      `);
      
      // Execute user code
      await window.pyodide.runPython(code);
      
      // Get output and any plots
      const output = await window.pyodide.runPython('sys.stdout.getvalue()');
      
      // Check if there's a plot
      let plotData = null;
      try {
        plotData = await window.pyodide.runPython(`
try:
    if plt.get_fignums():
        show_plot()
    else:
        None
except:
    None
        `);
      } catch {
        // No plot or plot error
      }
      
      // Restore stdout
      await window.pyodide.runPython('sys.stdout = _stdout');
      
      const executionTime = performance.now() - startTime;
      
      const result: ExecutionResult = {
        success: true,
        output: output || 'Code executed successfully (no output)',
        executionTime
      };
      
      // Add plot if available
      if (plotData && plotData !== 'None') {
        result.output += `\n\n[Plot generated]`;
        // You can extend this to show the actual plot
      }
      
      setExecutionResult(result);
      
      // Add to history
      setCodeHistory(prev => [...prev.slice(-4), {
        code,
        result,
        timestamp: Date.now()
      }]);
      
    } catch (error: any) {
      const executionTime = performance.now() - startTime;
      
      // Restore stdout on error
      try {
        await window.pyodide.runPython('sys.stdout = _stdout');
      } catch {}
      
      setExecutionResult({
        success: false,
        error: error.message || String(error),
        executionTime
      });
    } finally {
      setExecuting(false);
    }
  };

  // Generate code using AI
  const generateCode = async () => {
    if (!aiDescription.trim() || generatingCode || !apiUrl) return;
    
    setGeneratingCode(true);
    try {
      const response = await fetch(`${apiUrl}/generate-code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description: aiDescription,
          language: 'python'
        })
      });
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setCode(data.code);
        setAiDescription("");
      } else {
        alert('Failed to generate code');
      }
      
    } catch (error: any) {
      console.error('Code generation error:', error);
      alert(`Error generating code: ${error.message}`);
    } finally {
      setGeneratingCode(false);
    }
  };

  // Load example code
  const loadExample = (example: string) => {
    const examples = {
      plot: `import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Beautiful plot created!")`,

      data: `import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Score': np.random.randint(70, 100, 5),
    'City': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
}

df = pd.DataFrame(data)
print("Dataset created:")
print(df)
print(f"\\nSummary statistics:")
print(df.describe())
print(f"\\nAverage age: {df['Age'].mean():.1f}")
print(f"Highest score: {df['Score'].max()}")`,

      algorithm: `# Bubble Sort Algorithm
def bubble_sort(arr):
    n = len(arr)
    comparisons = 0
    
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    
    return arr, comparisons

# Test the algorithm
test_array = [64, 34, 25, 12, 22, 11, 90]
print(f"Original array: {test_array}")

sorted_array, comps = bubble_sort(test_array.copy())
print(f"Sorted array: {sorted_array}")
print(f"Number of comparisons: {comps}")

# Test with different sizes
import time
sizes = [10, 50, 100]
for size in sizes:
    arr = list(range(size, 0, -1))  # Worst case
    start = time.time()
    bubble_sort(arr)
    end = time.time()
    print(f"Size {size}: {(end-start)*1000:.2f}ms")`
    };
    
    setCode(examples[example as keyof typeof examples] || examples.plot);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">Python Code Playground</h1>
          <p className="text-gray-600 mb-4">
            Execute Python code directly in your browser using Pyodide. Supports NumPy, Matplotlib, Pandas, and more!
          </p>
          
          {/* Pyodide Status */}
          <div className="mb-4">
            {!pyodideReady && !pyodideLoading && (
              <button
                onClick={initializePyodide}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
              >
                Initialize Python Environment
              </button>
            )}
            
            {pyodideLoading && (
              <div className="text-blue-600">
                🔄 Loading Python environment... (this may take a minute)
              </div>
            )}
            
            {pyodideReady && (
              <div className="text-green-600">
                ✅ Python environment ready!
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Code Editor Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-black">Code Editor</h2>
              <div className="space-x-2">
                <button
                  onClick={() => loadExample('plot')}
                  className="bg-gray-500 text-white px-3 py-1 rounded text-sm hover:bg-gray-600"
                >
                  Plot Example
                </button>
                <button
                  onClick={() => loadExample('data')}
                  className="bg-gray-500 text-white px-3 py-1 rounded text-sm hover:bg-gray-600"
                >
                  Data Example
                </button>
                <button
                  onClick={() => loadExample('algorithm')}
                  className="bg-gray-500 text-white px-3 py-1 rounded text-sm hover:bg-gray-600"
                >
                  Algorithm Example
                </button>
              </div>
            </div>

            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="text-black w-full h-96 p-4 border border-gray-300 rounded font-mono text-sm resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter your Python code here..."
              style={{ fontFamily: 'Monaco, Consolas, "Lucida Console", monospace' }}
            />

            <div className="mt-4 space-y-4">
              {/* AI Code Generation */}
              <div className="border-t pt-4">
                <h3 className="font-semibold mb-2">🤖 AI Code Generator</h3>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={aiDescription}
                    onChange={(e) => setAiDescription(e.target.value)}
                    placeholder="Describe what code you want (e.g., 'create a bar chart of sales data')"
                    className="text-black flex-1 p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                    onKeyPress={(e) => e.key === 'Enter' && generateCode()}
                  />
                  <button
                    onClick={generateCode}
                    disabled={generatingCode || !apiUrl}
                    className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600 disabled:bg-gray-400"
                  >
                    {generatingCode ? '🔄' : '✨ Generate'}
                  </button>
                </div>
              </div>

              {/* Execute Button */}
              <button
                onClick={executeCode}
                disabled={!pyodideReady || executing}
                className="w-full bg-green-500 text-white py-3 rounded font-semibold hover:bg-green-600 disabled:bg-gray-400"
              >
                {executing ? '🔄 Executing...' : '▶️ Run Code'}
              </button>
            </div>
          </div>

          {/* Output Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Output</h2>
            
            <div 
              ref={outputRef}
              className="bg-gray-900 text-green-400 p-4 rounded h-96 overflow-y-auto font-mono text-sm"
            >
              {!executionResult && (
                <div className="text-gray-500">Output will appear here...</div>
              )}
              
              {executionResult && (
                <div>
                  {executionResult.success ? (
                    <div>
                      <div className="text-green-400 mb-2">
                        ✅ Executed successfully in {executionResult.executionTime?.toFixed(2)}ms
                      </div>
                      <pre className="whitespace-pre-wrap text-white">
                        {executionResult.output}
                      </pre>
                    </div>
                  ) : (
                    <div>
                      <div className="text-red-400 mb-2">
                        ❌ Execution failed in {executionResult.executionTime?.toFixed(2)}ms
                      </div>
                      <pre className="whitespace-pre-wrap text-red-300">
                        {executionResult.error}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Execution History */}
            {codeHistory.length > 0 && (
              <div className="mt-4">
                <h3 className="font-semibold mb-2">Recent Executions</h3>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {codeHistory.slice().reverse().map((entry, index) => (
                    <div 
                      key={entry.timestamp}
                      className="text-sm p-2 bg-gray-100 rounded cursor-pointer hover:bg-gray-200"
                      onClick={() => setCode(entry.code)}
                    >
                      <span className={entry.result.success ? 'text-green-600' : 'text-red-600'}>
                        {entry.result.success ? '✅' : '❌'}
                      </span>
                      <span className="ml-2 text-gray-600">
                        {new Date(entry.timestamp).toLocaleTimeString()}
                      </span>
                      <span className="ml-2 text-gray-800">
                        {entry.code.split('\n')[0].substring(0, 50)}...
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Information Section */}
       
      </div>
    </div>
  );
}