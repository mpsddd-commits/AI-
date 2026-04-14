const Header = () => {
  return (
    <header style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '20px 40px',
      borderBottom: '1px solid var(--glass-border)',
      background: 'rgba(5, 5, 16, 0.8)',
      backdropFilter: 'blur(10px)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '50%',
          background: 'var(--neon-cyan)',
          boxShadow: '0 0 15px var(--neon-cyan)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          color: 'black',
          fontWeight: 'bold',
          fontFamily: 'Orbitron'
        }}>
          AI
        </div>
        <h1 className="glow-text" style={{ margin: 0, fontSize: '1.5rem', color: 'var(--text-primary)' }}>
          NEURAL LINK TERMINAL
        </h1>
      </div>
      
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        fontFamily: 'Orbitron',
        fontSize: '0.8rem',
        color: 'var(--neon-pink)',
        textShadow: '0 0 5px var(--neon-pink)'
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: 'var(--neon-pink)',
          animation: 'pulseGlow 2s infinite'
        }} />
        SYSTEM ONLINE
      </div>
    </header>
  );
};

export default Header;
