import axios from "axios"

export const api = axios.create({
  baseURL: import.meta.env.VITE_APP_FASTAPI_URL || "http://localhost:8000",
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
})

// export const fetchPosts = async () => {
//   try {
//     const response = await api.post("/chat", { 
//       // AI가 도구 호출을 잊지 않도록 명확하게 지시
//       prompt: "데이터베이스에서 모든 게시글 목록을 list_all_posts 도구를 사용하여 조회하고 JSON으로 반환해줘." 
//     });
    
//     // 백엔드 response.data.response가 배열인지 확인
//     let posts = response.data.response;

//     // 만약 AI가 { "data": [...] } 형태로 감싸서 보냈을 경우를 대비
//     if (posts && posts.data) {
//       posts = posts.data;
//     }

//     return Array.isArray(posts) ? posts : [];
//   } catch (error) {
//     console.error("Failed to fetch posts:", error);
//     return [];
//   }
// };


export const fetchPosts = async () => {
  console.log("🚀 백엔드 호출 시작: http://127.0.0.1:8000/posts");
  try {
    const response = await api.get("/posts");
    console.log("✅ 백엔드 응답 성공:", response.data);
    return response.data;
  } catch (error) {
    console.error("❌ 백엔드 호출 에러:", error.message);
    // 에러 상세 정보 확인
    if (error.response) {
      console.error("Data:", error.response.data);
      console.error("Status:", error.response.status);
    }
    return [];
  }
};


// export const fetchPosts = async () => {
//   try {
//     // 1. /chat 대신 /posts 엔드포인트로 GET 요청을 보냄
//     const response = await api.get("/posts");
    
//     // 2. 서버에서 바로 리스트를 반환하므로 response.data가 곧 posts임
//     const posts = response.data;

//     return Array.isArray(posts) ? posts : [];
//   } catch (error) {
//     console.error("데이터베이스에서 목록을 직접 가져오는데 실패했습니다:", error);
//     return [];
//   }
// };


// export const createPost = async (prompt) => { // formData 대신 prompt(문자열)를 받음
//   try {
//     const response = await api.post("/chat", {
//       prompt: prompt  // 사용자가 입력한 텍스트 그대로 전송
//     });
    
//     // 백엔드에서 생성된 후 반환하는 게시글 데이터
//     return response.data.response;
//   } catch (error) {
//     console.error("Error creating post:", error);
//     throw error;
//   }
// };

export const createPost = async (prompt) => {
  try {
    const response = await api.post("/chat", { prompt });
    
    // response.data가 null이거나 response 키가 없을 경우를 대비
    if (!response.data || response.data.response === undefined) {
      console.warn("AI 응답 형식이 불완전합니다:", response.data);
      return null; 
    }
    
    return response.data.response;
  } catch (error) {
    console.error("API 호출 실패:", error);
    throw error;
  }
};