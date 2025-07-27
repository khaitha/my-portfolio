import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const { user_message, has_active_code, code_context, conversation_context } = body
    
    if (!user_message) {
      return NextResponse.json({ error: 'User message is required' }, { status: 400 })
    }

    console.log('Forwarding code action analysis request')

    // Build API URL
    const apiUrl = process.env.NODE_ENV === 'production' 
      ? `https://goldfish-app-84zag.ondigitalocean.app/my-portfolio-portfolio-api/analyze-code-action`
      : `http://localhost:8000/analyze-code-action`

    console.log('Calling API URL:', apiUrl)

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        user_message, 
        has_active_code, 
        code_context, 
        conversation_context 
      }),
    })

    console.log('Analyze-code-action API response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Analyze-code-action API error response:', errorText)
      throw new Error(`Code action analysis service failed: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    console.log('Code action analysis successful')
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
