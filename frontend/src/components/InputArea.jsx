import { useState, useRef, useEffect } from 'react'
import './InputArea.css'

function InputArea({ onSendMessage, disabled }) {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`
    }
  }, [text])

  const handleSubmit = (e) => {
    e?.preventDefault()
    if (!text.trim() || disabled) return
    onSendMessage(text.trim())
    setText('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="input-area-wrapper">
      <form className="input-form" onSubmit={handleSubmit}>
        <div className={`input-container ${disabled ? 'disabled' : ''}`}>
          <textarea
            ref={textareaRef}
            className="input-textarea"
            placeholder="Ask a question about Agentic AI concepts, tools, or architectures..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            rows={1}
          />
          <button
            type="submit"
            className="send-button"
            disabled={disabled || !text.trim()}
            aria-label="Send message"
          >
            <svg
              className="send-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </form>
      <div className="input-footer">
        <span>Hybrid Dense (Gemini) + Sparse (BM25) • RRF Fusion • Groq LLM</span>
        <span><kbd>Enter</kbd> to send • <kbd>Shift + Enter</kbd> for new line</span>
      </div>
    </div>
  )
}

export default InputArea
