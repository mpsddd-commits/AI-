## FAISS

- `faiss-cpu`: CPU 버전
- `faiss-gpu`: GPU 버전

## Embedding Model -rag함

- [mxbai-embed-large:latest](https://ollama.com/library/mxbai-embed-large)
- [nomic-embed-text:v1.5](https://ollama.com/library/nomic-embed-text) 
- [nomic-embed-text-v2-moe:latest](https://ollama.com/library/nomic-embed-text-v2-moe) => 멀티모달용
- [qwen3-embedding:8b](https://ollama.com/library/qwen3-embedding)

## LLM Model

- [llama3.2:3b](https://ollama.com/library/llama3.2)
- [gemma3:4b](https://ollama.com/library/gemma3)
- [gemma4:e4b](https://ollama.com/library/gemma4) => 최근 버젼
- [gpt-oss:20b](https://ollama.com/library/gpt-oss)
- [qwen3:8b](https://ollama.com/library/qwen3)

## Ollama App (모델 확인)

```bash
ollama list
```

## 터미널 실행

```bash
ollama serve
```

## `.env` 설정

```bash
OLLAMA_BASE_URL=http://localhost:11434
```
