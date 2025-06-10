import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { query, num_results = 5 } = await request.json()

    if (!query) {
      return NextResponse.json({ error: 'No query provided' }, { status: 400 })
    }

    console.log('Forwarding search request:', { query, num_results })

    // Forward to FastAPI search service
    const response = await fetch('http://localhost:8001/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        query,
        num_results 
      }),
    })

    console.log('FastAPI response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('FastAPI error response:', errorText)
      throw new Error(`Search service failed: ${response.status}`)
    }

    const data = await response.json()
    console.log('Search successful, returning data')
    return NextResponse.json(data)

  } catch (error) {
    console.error('Search API error:', error)
    
    if (error instanceof Error && error.message.includes('ECONNREFUSED')) {
      return NextResponse.json(
        { error: 'Search service unavailable. Make sure the FastAPI service is running on port 8001.' },
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { error: 'Search failed' },
      { status: 500 }
    )
  }
}