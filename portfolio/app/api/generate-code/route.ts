import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const { description, language = 'python' } = body
    
    if (!description) {
      return NextResponse.json({ error: 'Description is required' }, { status: 400 })
    }

    console.log('Forwarding code generation request')

    // Build API URL
    const apiUrl = process.env.NODE_ENV === 'production' 
      ? `https://goldfish-app-84zag.ondigitalocean.app/my-portfolio-portfolio-api/generate-code`
      : `http://localhost:8000/generate-code`

    console.log('Calling API URL:', apiUrl)

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ description, language }),
    })

    console.log('Generate-code API response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Generate-code API error response:', errorText)
      throw new Error(`Code generation service failed: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    console.log('Code generation successful')
    return NextResponse.json(data)

  } catch (error) {
    console.error('API error:', error)
    
    // More detailed error logging for production debugging
    if (error instanceof Error) {
      console.error('Error message:', error.message)
      console.error('Error stack:', error.stack)
    }
    
    if (error instanceof Error && error.message.includes('ECONNREFUSED')) {
      return NextResponse.json(
        { error: 'Backend service unavailable. Please try again later.' },
        { status: 503 }
      )
    }
    
    if (error instanceof Error && error.message.includes('fetch')) {
      return NextResponse.json(
        { error: 'Failed to connect to code generation service. Please try again.' },
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { error: `Request failed: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    )
  }
}
