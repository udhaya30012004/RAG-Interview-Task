import { useEffect, useRef } from 'react'
import './ChatContainer.css'
import Message from './Message'
import EmptyState from './EmptyState'

function ChatContainer({ messages, onSuggestionClick }) {
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return <EmptyState onSuggestionClick={onSuggestionClick} />
  }

  return (
    <div className="chat-container">
      {messages.map((message) => (
        <Message key={message.id} message={message} />
      ))}
      <div ref={chatEndRef} />
    </div>
  )
}

export default ChatContainer
