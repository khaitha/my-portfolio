import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 })
    }

    // Check file type
    if (!file.type.includes('pdf')) {
      return NextResponse.json({ error: 'Only PDF files are allowed' }, { status: 400 })
    }

    // Check file size (50MB limit)
    if (file.size > 50 * 1024 * 1024) {
      return NextResponse.json({ error: 'File too large. Maximum size is 50MB.' }, { status: 400 })
    }

    console.log('Uploading file:', file.name, 'Size:', file.size, 'bytes')

    // Build API URL
    const apiUrl = process.env.NODE_ENV === 'production' 
      ? `https://goldfish-app-84zag.ondigitalocean.app/my-portfolio-portfolio-api/upload`
      : `http://localhost:8000/upload`

    console.log('Calling upload API URL:', apiUrl)

    // Forward the file to the backend API
    const backendFormData = new FormData()
    backendFormData.append('file', file)

    const response = await fetch(apiUrl, {
      method: 'POST',
      body: backendFormData,
    })

    console.log('Upload API response status:', response.status)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Upload API error response:', errorText)
      throw new Error(`Upload service failed: ${response.status} - ${errorText}`)
    }

    const data = await response.json()
    console.log('Upload successful, returning data')
    return NextResponse.json(data)

  } catch (error: any) {
    console.error('Upload API error:', error)
    return NextResponse.json(
      { error: error.message || 'Upload failed' },
      { status: 500 }
    )
  }
}
