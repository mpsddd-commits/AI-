const PostDetail = ({ post, onBack }) => {
  if (!post) return null;

  return (
    <div className="glass-panel" style={{ padding: '30px', animation: 'pulseGlow 2s infinite' }}>
      <button 
        onClick={onBack} 
        className="cyber-button" 
        style={{ marginBottom: '30px', padding: '10px 20px' }}
      >
        &lt; RETURN TO MAIN SEQUENCE
      </button>

      <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '15px' }}>
        <h2 style={{ color: 'var(--text-primary)', margin: 0 }}>
          {post.title}
        </h2>
        {post.ai_tags && (
          <span style={{
            border: '1px solid var(--neon-pink)',
            padding: '3px 8px',
            borderRadius: '3px',
            fontSize: '0.8rem',
            color: 'var(--neon-pink)',
            fontFamily: 'Orbitron',
            textShadow: '0 0 5px rgba(255, 0, 60, 0.5)'
          }}>
            {post.ai_tags}
          </span>
        )}
      </div>

      <div style={{ 
        color: 'var(--neon-purple)', 
        marginBottom: '30px', 
        fontFamily: 'Orbitron',
        fontSize: '0.9rem',
        borderBottom: '1px solid var(--glass-border)',
        paddingBottom: '15px'
      }}>
        OPERATIVE NAME: <span style={{ color: 'var(--text-primary)' }}>{post.name}</span> // 
        TIMESTAMP: <span style={{ color: 'var(--text-primary)' }}>{post.created_at ? new Date(post.created_at).toLocaleString() : 'JUST NOW'}</span>
      </div>

      <div style={{ 
        lineHeight: '1.8', 
        color: 'var(--text-primary)',
        fontFamily: 'Rajdhani',
        fontSize: '1.2rem',
        whiteSpace: 'pre-wrap',
        background: 'rgba(0, 0, 0, 0.4)',
        padding: '25px',
        borderRadius: '5px',
        border: '1px solid var(--glass-border)',
        boxShadow: 'inset 0 0 20px rgba(0, 0, 0, 0.8)'
      }}>
        {post.content}
      </div>
    </div>
  );
};

export default PostDetail;
