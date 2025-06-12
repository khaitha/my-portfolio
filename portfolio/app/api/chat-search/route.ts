import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    // Check if this is a chat-search request (has 'messages' field)
    const isChatSearch = 'messages' in body

    if (!isChatSearch) {
      return NextResponse.json({ error: 'Invalid request format' }, { status: 400 })
    }

    // Chat-search request
    const { messages, search_when_needed = true, num_search_results = 5 } = body
    
    if (!messages || !Array.isArray(messages)) {
      return NextResponse.json({ error: 'No messages provided' }, { status: 400 })
    }

    const payload = { messages, search_when_needed, num_search_results }
    console.log('Forwarding chat-search request with', messages.length, 'messages')

    // Build API URL
    const apiUrl = process.env.NODE_ENV === 'production' 
      ? `https://goldfish-app-84zag.ondigitalocean.app/my-portfolio-portfolio-api/chat-search`
      : `http://localhost:8000/chat-search`

    console.log('Calling API URL:', apiUrl)

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })

    console.log('Chat-search API response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Chat-search API error response:', errorText)
      throw new Error(`Chat-search service failed: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    console.log('Chat-search successful, returning data')
    return NextResponse.json(data)

  } catch (error) {
    console.error('API error:', error)
    
    if (error instanceof Error && error.message.includes('ECONNREFUSED')) {
      return NextResponse.json(
        { error: 'Service unavailable. Please ensure the backend service is running.' },
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { error: `Request failed: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    )
  }
}