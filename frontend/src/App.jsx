import { useState, useEffect, useRef } from 'react'
import './App.css'
import Header from './components/Header'
import ChatContainer from './components/ChatContainer'
import InputArea from './components/InputArea'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const threadId = useRef('session-' + Date.now())

  const sendMessage = async (messageText) => {
    if (!messageText.trim() || isLoading) return

    setIsLoading(true)

    // Add user message
    const userMessage = {
      id: Date.now(),
      role: 'user',
      text: messageText,
      timestamp: new Date()
    }
    setMessages(prev => [...prev, userMessage])

    // Add loading message
    const loadingId = Date.now() + 1
    setMessages(prev => [...prev, { id: loadingId, role: 'loading' }])

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: messageText,
          thread_id: threadId.current
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      // Remove loading message and add assistant response
      setMessages(prev => {
        const filtered = prev.filter(msg => msg.id !== loadingId)
        return [...filtered, {
          id: Date.now() + 2,
          role: 'assistant',
          data: data,
          timestamp: new Date()
        }]
      })

    } catch (error) {
      console.error('Error:', error)

      // Remove loading and add error message
      setMessages(prev => {
        const filtered = prev.filter(msg => msg.id !== loadingId)
        return [...filtered, {
          id: Date.now() + 2,
          role: 'assistant',
          data: {
            answer: `Sorry, I encountered an error: ${error.message}. Please make sure the API server is running at ${API_BASE_URL}`,
            status: 'error',
            sources: [],
            metadata: {}
          },
          timestamp: new Date()
        }]
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleSuggestionClick = (question) => {
    sendMessage(question)
  }

  return (
    <div className="app">
      <Header />
      <ChatContainer
        messages={messages}
        onSuggestionClick={handleSuggestionClick}
      />
      <InputArea
        onSendMessage={sendMessage}
        disabled={isLoading}
      />
    </div>
  )
}

export default App
