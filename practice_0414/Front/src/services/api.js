import axios from "axios"

export const api = axios.create({
  baseURL: import.meta.env.VITE_APP_FASTAPI_URL || "http://127.0.0.1:8000",
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
})

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