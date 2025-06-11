import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { query, num_results = 5 } = await request.json()

    if (!query) {
      return NextResponse.json({ error: 'No query provided' }, { status: 400 })
    }

    console.log('Forwarding search request:', { query, num_results })

    // Fix the URL - your search service runs on port 8000, not 3000
    const apiUrl = process.env.NODE_ENV === 'production' 
      ? "https://goldfish-app-84zag.ondigitalocean.app/my-portfolio-portfolio-api'
      : 'http://localhost:8000/search'  // Changed from port 3000 to 8000

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        query,
        num_results 
      }),
    })

    console.log('Search API response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Search API error response:', errorText)
      throw new Error(`Search service failed: ${response.status}`)
    }

    const data = await response.json()
    console.log('Search successful, returning data')
    return NextResponse.json(data)

  } catch (error) {
    console.error('Search API error:', error)
    
    if (error instanceof Error && error.message.includes('ECONNREFUSED')) {
      return NextResponse.json(
        { error: 'Search service unavailable. Please ensure the search service is running on port 8000.' },
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { error: 'Search failed' },
      { status: 500 }
    )
  }
}