import './EmptyState.css'

const SUGGESTIONS = [
  {
    title: 'Core Fundamentals',
    question: 'What is an AI Agent and how does it differ from a traditional LLM?',
    icon: '💡'
  },
  {
    title: 'Agent Architecture',
    question: 'What are the core components and key design patterns of an Agentic AI system?',
    icon: '⚙️'
  },
  {
    title: 'Multi-Agent Systems',
    question: 'What are Multi-Agent Systems and how do they work?',
    icon: '🤝'
  }
]

function EmptyState({ onSuggestionClick }) {
  return (
    <div className="empty-state">
      <div className="empty-hero">
        <div className="empty-badge">Hybrid RAG System</div>
        <h2 className="empty-title">Ask anything about Agentic AI</h2>
        <p className="empty-description">
          Powered by Pinecone vector search, BM25 sparse retrieval, Reciprocal Rank Fusion, and LangGraph orchestration.
        </p>
      </div>

      <div className="empty-features">
        <div className="feature-pill">
          <span className="feature-dot dense"></span>
          <span>Dense Semantic (Gemini)</span>
        </div>
        <div className="feature-pill">
          <span className="feature-dot sparse"></span>
          <span>Sparse Keyword (BM25)</span>
        </div>
        <div className="feature-pill">
          <span className="feature-dot fusion"></span>
          <span>RRF Merged (k=60)</span>
        </div>
        <div className="feature-pill">
          <span className="feature-dot graph"></span>
          <span>Adaptive Retry Graph</span>
        </div>
      </div>

      <div className="suggestions-container">
        <div className="suggestions-label">Suggested Questions</div>
        <div className="suggestions-grid">
          {SUGGESTIONS.map((item, index) => (
            <button
              key={index}
              className="suggestion-card"
              onClick={() => onSuggestionClick(item.question)}
            >
              <div className="suggestion-card-header">
                <span className="suggestion-icon">{item.icon}</span>
                <span className="suggestion-category">{item.title}</span>
              </div>
              <div className="suggestion-text">{item.question}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default EmptyState
