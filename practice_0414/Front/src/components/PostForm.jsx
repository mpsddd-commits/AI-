import { useState } from 'react';

const PostForm = ({ onPostCreated }) => {
  const [prompt, setPrompt] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    
    setIsSubmitting(true);
    await onPostCreated(prompt);
    
    setPrompt('');
    setIsSubmitting(false);
  };

  return (
    <div className="glass-panel" style={{ padding: '24px', marginBottom: '30px' }}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <span style={{ fontSize: '1.5rem', color: 'var(--neon-purple)', fontFamily: 'Orbitron' }}>&gt;</span>
          <textarea
            className="cyber-input"
            placeholder="이름은 [이름], 제목은 [제목], 내용은 [내용]으로 게시글 적어줘"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            required
            autoComplete="off"
            rows={3}
            style={{ flex: 1, padding: '15px', fontSize: '1.1rem', resize: 'vertical' }}
          />
          <button 
            type="submit" 
            className="cyber-button"
            disabled={isSubmitting}
            style={{ padding: '15px 30px', fontSize: '1rem', height: '100%' }}
          >
            {isSubmitting ? 'PROCESSING...' : 'EXECUTE'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default PostForm;
