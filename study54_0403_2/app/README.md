## GGUF 변환 (llama.cpp 사용)

1. llama.cpp 설치
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- `config.json` 파일 생성 (모델_폴더_경로에 추가)
```json
{
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "model_type": "gpt2",
  "vocab_size": 50257,
  "n_positions": 128,
  "n_ctx": 128,
  "n_embd": 768,
  "n_layer": 12,
  "n_head": 12,
  "n_inner": 3072,
  "activation_function": "gelu_new",
  "resid_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "attn_pdrop": 0.1,
  "layer_norm_epsilon": 1e-05,
  "initializer_range": 0.02,
  "bos_token_id": 50256,
  "eos_token_id": 50256
}
```


2. 모델 변환:
```bash
python convert_hf_to_gguf.py  <모델_폴더_경로> --outfile my_model.gguf --outtype f16
```

3. Ollama Modelfile 생성
```dockerfile
# 1. 변환된 GGUF 파일 경로
FROM ./model.gguf

# 2. 모델 매개변수 설정
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"

# 3. 템플릿 설정 (GPT-2 스타일)
TEMPLATE """
{{ .Prompt }}
"""

# 4. 시스템 메시지
SYSTEM "You are a helpful AI assistant trained on custom data."
```

4. Ollama 모델 등록 및 실행
모델 생성:
```bash
ollama create my-custom-gpt -f Modelfile
```

모델 실행:
```bash
ollama run my-custom-gpt
```
