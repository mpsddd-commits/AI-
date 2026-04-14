import { useState, useEffect } from 'react';
import Header from './components/Header';
import PostForm from './components/PostForm';
import PostList from './components/PostList';
import PostDetail from './components/PostDetail';
import { fetchPosts, createPost } from './services/api';
import './index.css';

function App() {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPost, setSelectedPost] = useState(null);

  useEffect(() => {
    loadPosts();
  }, []);

  const loadPosts = async () => {
    setLoading(true);
    try {
      const data = await fetchPosts();
      setPosts(data);
    } catch (error) {
      console.error('Error fetching data', error);
    } finally {
      setLoading(false);
    }
  };

const handlePostCreated = async (promptText) => {
  try {
    await createPost(promptText); // 명령 실행
  } catch (error) {
    console.error('명령 실행 중 에러가 발생했으나 목록을 새로고침합니다.', error);
  } finally {
    // 성공하든 실패하든 DB에서 최신 목록을 가져와서 화면을 동기화합니다.
    await loadPosts();
    setSelectedPost(null);
  }
};


// const handlePostCreated = async (promptText) => { // 여기서 promptText를 받음
//   try {
//     const newPost = await createPost(promptText);
    
//     // 만약 서버 응답이 배열이라면 합치고, 객체라면 배열에 추가
//     if (Array.isArray(newPost)) {
//       setPosts(prev => [...newPost, ...prev]);
//     } else {
//       setPosts(prev => [newPost, ...prev]);
//     }
    
//     // 목록을 최신화하기 위해 다시 불러오는 것도 좋은 방법입니다.
//     loadPosts(); 
//   } catch (error) {
//     console.error('Error creating post', error);
//   }
// };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />
      
      <main style={{ 
        flex: 1, 
        maxWidth: '900px', 
        width: '100%', 
        margin: '0 auto', 
        padding: '40px 20px' 
      }}>
        
        {loading ? (
          <div style={{ textAlign: 'center', margin: '50px 0', color: 'var(--neon-cyan)', fontFamily: 'Orbitron' }}>
            <div style={{ display: 'inline-block', width: '20px', height: '20px', border: '2px solid var(--neon-cyan)', borderRadius: '50%', borderTopColor: 'transparent', animation: 'spin 1s linear infinite' }}></div>
            <div style={{ marginTop: '10px' }}>ESTABLISHING SYSTEM CONNECTION...</div>
            <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
          </div>
        ) : selectedPost ? (
          <PostDetail 
            post={selectedPost} 
            onBack={() => setSelectedPost(null)} 
          />
        ) : (
          <>
            <PostForm onPostCreated={handlePostCreated} />
            <PostList posts={posts} onPostClick={(post) => setSelectedPost(post)} />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
