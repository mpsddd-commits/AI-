const PostList = ({ posts, onPostClick }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <h3 style={{ 
        color: 'var(--text-secondary)', 
        borderBottom: '1px solid var(--glass-border)', 
        paddingBottom: '10px',
        fontFamily: 'Orbitron'
      }}>
        DATA LOG_
      </h3>
      
      {posts.length === 0 ? (
        <div style={{ color: 'var(--text-secondary)', fontStyle: 'italic', padding: '20px' }}>
          NO RECORDS FOUND IN CURRENT SECTOR...
        </div>
      ) : (
        posts.map((post, idx) => (
          <div 
            key={post.id || `post-${idx}`}  // post.id가 없으면 idx를 사용
            className="glass-panel"
            style={{ 
              padding: '20px', 
              position: 'relative',
            overflow: 'hidden',
            transition: 'transform 0.2s',
            cursor: 'pointer',
          }}
          onClick={() => onPostClick(post)}
          onMouseOver={(e) => { e.currentTarget.style.transform = 'translateX(10px) scale(1.01)'; }}
            onMouseOut={(e) => { e.currentTarget.style.transform = 'translateX(0) scale(1)'; }}
          >
            {/* Edge decorative line */}
            <div style={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: '4px',
              backgroundColor: 'var(--neon-cyan)',
              boxShadow: '0 0 10px var(--neon-cyan)'
            }} />
            
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '15px' }}>
              <div>
                <h4 style={{ 
                  margin: '0 0 5px 0', 
                  fontSize: '1.2rem', 
                  color: 'var(--text-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px'
                }}>
                  {/* 여기에 ID 표시 추가: #001 스타일로 표시 */}
                  <span style={{ 
                    color: 'var(--neon-cyan)', 
                    fontSize: '0.9rem', 
                    fontFamily: 'Orbitron',
                    opacity: 0.8 
                  }}>
                    #{String(post.id).padStart(3, '0')}
                  </span>
                  
                  {post.title}
                </h4>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  ORIGIN: <span style={{ color: 'var(--neon-purple)' }}>{post.name}</span> // 
                  TIME: {post.created_at ? new Date(post.created_at).toLocaleString() : 'JUST NOW'}
                </div>
              </div>
              
              {post.ai_tags && (
                <div style={{
                  border: '1px solid var(--neon-pink)',
                  padding: '3px 8px',
                  borderRadius: '3px',
                  fontSize: '0.7rem',
                  color: 'var(--neon-pink)',
                  fontFamily: 'Orbitron',
                  textShadow: '0 0 5px rgba(255, 0, 60, 0.5)'
                }}>
                  {post.ai_tags}
                </div>
              )}
            </div>
            
            <div style={{ 
              lineHeight: '1.6', 
              color: 'rgba(224, 251, 252, 0.9)',
              fontFamily: 'Rajdhani',
              fontSize: '1.05rem',
              whiteSpace: 'pre-wrap'
            }}>
              {post.content}
            </div>
          </div>
        ))
      )}
    </div>
  );
};

export default PostList;
