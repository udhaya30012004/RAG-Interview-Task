import { useState, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import './Message.css'

function cleanMarkdown(rawText) {
  if (!rawText) return ''
  let text = rawText

  // Fix mismatched markdown asterisks e.g. *Agentic AI** -> **Agentic AI**
  text = text.replace(/(^|[^\*])\*([^\*\n]+)\*\*(?!\*)/g, '$1**$2**')
  text = text.replace(/(^|[^\*])\*\*([^\*\n]+)\*(?!\*)/g, '$1**$2**')

  return text
}

function Message({ message }) {
  const [showSources, setShowSources] = useState(false)
  const [showScores, setShowScores] = useState(false)
  const [expandedChunkId, setExpandedChunkId] = useState(null)
  const [copied, setCopied] = useState(false)

  if (message.role === 'loading') {
    return (
      <div className="message-wrapper assistant loading">
        <div className="avatar assistant-avatar">🤖</div>
        <div className="message-content">
          <div className="loading-container">
            <div className="loading-steps">
              <div className="loading-pulse"></div>
              <span className="loading-text">Performing Dense + Sparse Hybrid Retrieval & RRF Fusion...</span>
            </div>
            <div className="skeleton-line" style={{ width: '80%' }}></div>
            <div className="skeleton-line" style={{ width: '95%' }}></div>
            <div className="skeleton-line" style={{ width: '60%' }}></div>
          </div>
        </div>
      </div>
    )
  }

  if (message.role === 'user') {
    return (
      <div className="message-wrapper user">
        <div className="message-content">
          <div className="user-message-bubble">
            <p className="user-message-text">{message.text}</p>
          </div>
          <span className="message-timestamp">
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>
        <div className="avatar user-avatar">👤</div>
      </div>
    )
  }

  // Assistant Message
  const data = message.data || {}
  const { answer = '', sources = [], status = 'success', metadata = {} } = data
  const {
    retrieval_attempt = 1,
    dense_count = 0,
    sparse_count = 0,
    top_rrf_score = 0,
    dense_scores = [],
    sparse_scores = [],
    rrf_scores = []
  } = metadata

  const formattedAnswer = useMemo(() => cleanMarkdown(answer), [answer])

  const handleCopy = () => {
    // Copy plain text without formatting asterisks
    const plainText = answer.replace(/\*\*/g, '').replace(/\*/g, '')
    navigator.clipboard.writeText(plainText)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const toggleChunk = (chunkId) => {
    setExpandedChunkId(prev => (prev === chunkId ? null : chunkId))
  }

  const getStatusBadge = () => {
    switch (status) {
      case 'success':
      case 'answered':
        return <span className="status-badge success">✓ Hybrid Retrieval Verified</span>
      case 'retry_success':
        return <span className="status-badge retry">🔄 Adaptive Retry (Attempt {retrieval_attempt})</span>
      case 'fallback':
        return <span className="status-badge fallback">⚠️ Out-of-Context Fallback</span>
      case 'invalid_query':
        return <span className="status-badge invalid">⚠️ Invalid Query</span>
      case 'error':
        return <span className="status-badge error">❌ System Error</span>
      default:
        return <span className="status-badge">{status}</span>
    }
  }

  return (
    <div className="message-wrapper assistant">
      <div className="avatar assistant-avatar">🤖</div>
      <div className="message-content">
        <div className="assistant-card">
          <div className="card-top-bar">
            {getStatusBadge()}
            <div className="top-bar-actions">
              <button
                className="action-btn"
                onClick={handleCopy}
                title="Copy answer"
              >
                {copied ? '✓ Copied' : '📋 Copy'}
              </button>
            </div>
          </div>

          {/* Formatted Answer without raw asterisks */}
          <div className="answer-body">
            <ReactMarkdown
              components={{
                h1: ({ children }) => <h2 className="answer-heading h1">{children}</h2>,
                h2: ({ children }) => <h3 className="answer-heading h2">{children}</h3>,
                h3: ({ children }) => <h4 className="answer-heading h3">{children}</h4>,
                p: ({ children }) => <p className="answer-p">{children}</p>,
                ul: ({ children }) => <ul className="answer-ul">{children}</ul>,
                ol: ({ children }) => <ol className="answer-ol">{children}</ol>,
                li: ({ children }) => <li className="answer-li">{children}</li>,
                strong: ({ children }) => <strong className="answer-strong">{children}</strong>,
                em: ({ children }) => <em className="answer-em">{children}</em>,
                code: ({ inline, children }) => (
                  inline
                    ? <code className="answer-inline-code">{children}</code>
                    : <pre className="answer-code-block"><code>{children}</code></pre>
                )
              }}
            >
              {formattedAnswer}
            </ReactMarkdown>
          </div>

          {/* Quick Metrics Bar */}
          <div className="metrics-bar">
            <div className="metric-item">
              <span className="metric-label">Dense Matches:</span>
              <span className="metric-val">{dense_count}</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Sparse Matches:</span>
              <span className="metric-val">{sparse_count}</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Fused Top RRF:</span>
              <span className="metric-val">
                {typeof top_rrf_score === 'number' ? top_rrf_score.toFixed(4) : top_rrf_score}
              </span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Attempt:</span>
              <span className="metric-val">{retrieval_attempt}</span>
            </div>
          </div>

          {/* Score Insights & Sources Toggles */}
          <div className="details-toggle-row">
            {sources.length > 0 && (
              <button
                className={`toggle-section-btn ${showSources ? 'active' : ''}`}
                onClick={() => setShowSources(!showSources)}
              >
                <span>📚 Sources ({sources.length} Chunks)</span>
                <span className="arrow-icon">{showSources ? '▲' : '▼'}</span>
              </button>
            )}

            {(dense_scores.length > 0 || sparse_scores.length > 0 || rrf_scores.length > 0) && (
              <button
                className={`toggle-section-btn ${showScores ? 'active' : ''}`}
                onClick={() => setShowScores(!showScores)}
              >
                <span>📊 Score Breakdown</span>
                <span className="arrow-icon">{showScores ? '▲' : '▼'}</span>
              </button>
            )}
          </div>

          {/* Sources Section */}
          {showSources && sources.length > 0 && (
            <div className="sources-container">
              <div className="sources-header">Retrieved Context Chunks (Ranked by RRF)</div>
              <div className="sources-list">
                {sources.map((src, index) => {
                  const isExpanded = expandedChunkId === src.chunk_id
                  return (
                    <div key={src.chunk_id || index} className="source-card">
                      <div
                        className="source-card-header"
                        onClick={() => toggleChunk(src.chunk_id)}
                      >
                        <div className="source-meta">
                          <span className="source-rank">#{index + 1}</span>
                          <span className="source-page">Page {src.page_num}</span>
                          <span className="source-label">{src.label || src.chunk_id}</span>
                        </div>
                        <span className="expand-indicator">{isExpanded ? 'Collapse' : 'View Text'}</span>
                      </div>
                      {isExpanded && (
                        <div className="source-text">
                          <p>{src.text}</p>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Score Breakdown Section */}
          {showScores && (
            <div className="scores-container">
              <div className="scores-header">Dense vs Sparse vs Fused Score Details</div>
              <div className="scores-table-wrapper">
                <table className="scores-table">
                  <thead>
                    <tr>
                      <th>Chunk ID</th>
                      <th>Page</th>
                      <th>Dense Score</th>
                      <th>BM25 Score</th>
                      <th>RRF Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rrf_scores.map((item, idx) => {
                      const dItem = dense_scores.find(d => d.chunk_id === item.chunk_id)
                      const sItem = sparse_scores.find(s => s.chunk_id === item.chunk_id)
                      return (
                        <tr key={idx}>
                          <td className="code-cell">{item.chunk_id}</td>
                          <td>{item.page_num}</td>
                          <td>{dItem ? dItem.score.toFixed(4) : '-'}</td>
                          <td>{sItem ? sItem.score.toFixed(4) : '-'}</td>
                          <td className="highlight-cell">{item.rrf_score.toFixed(5)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
        <span className="message-timestamp assistant-ts">
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  )
}

export default Message
