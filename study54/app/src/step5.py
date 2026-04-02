from settings import settings
import os
import torch
import tiktoken
from src.train_model import GPTModel

# 1. 텍스트 생성 핵심 함수
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
  """
  Args:
    idx: 현재까지 입력된 토큰 ID 리스트 (입력 문맥)
    max_new_tokens: 생성할 최대 토큰 수
    context_size: 모델이 한 번에 볼 수 있는 최대 문맥 길이 (Position Embedding 크기)
    temperature: 값이 높을수록 창의적이고 낮을수록 결정론적인(안전한) 문장 생성
    top_k: 확률 상위 K개 토큰만 후보로 사용 (노이즈 제거)
  """
  for _ in range(max_new_tokens):
    # 모델의 처리 능력을 넘지 않도록 최근 context_size만큼의 토큰만 추출
    idx_cond = idx[:, -context_size:]
    
    with torch.no_grad(): # 추론 시에는 기울기 계산 제외
      logits = model(idx_cond)
    
    # 마지막 타임스텝의 결과(다음에 올 토큰의 점수)만 가져옴
    logits = logits[:, -1, :]

    # Top-K 필터링: 확률이 낮은 토큰들을 후보에서 제외 (무한대 음수값 처리)
    if top_k is not None:
      top_logits, _ = torch.topk(logits, top_k)
      min_val = top_logits[:, -1]
      logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

    # 샘플링 방식 결정
    if temperature > 0.0:
      # 온도를 적용하여 확률 분포를 조절 (Softmax 적용 전 나눗셈)
      logits = logits / temperature
      probs = torch.softmax(logits, dim=-1)
      # 확률 분포에 따라 랜덤하게 토큰 선택 (창의성 부여)
      idx_next = torch.multinomial(probs, num_samples=1)
    else:
      # 온도 0일 경우 가장 확률이 높은 토큰을 단순히 선택 (Greedy Search)
      idx_next = torch.argmax(logits, dim=-1, keepdim=True)

    # 문장 종료 토큰(EOS)이 생성되면 반복 중단
    if idx_next == eos_id:
      break

    # 새로 생성된 토큰을 기존 시퀀스 뒤에 이어 붙임
    idx = torch.cat((idx, idx_next), dim=1)

  return idx

# 2. 다음 토큰 예측 확률 테스트 함수
def test1(model, tokenizer, device, keywords:str = "Harry Potter", filename:str = ""):
  """단일 입력을 넣었을 때 모델이 예측하는 '다음 토큰' 후보 10개를 출력"""
  idx = tokenizer.encode(keywords)
  idx = torch.tensor(idx).unsqueeze(0).to(device)
  
  with torch.no_grad():
    logits = model(idx)
  logits = logits[:, -1, :]

  # 가장 확률이 높은 상위 10개 토큰과 점수를 확인
  top_logits, top_indices = torch.topk(logits, 10) 
  for p, i in zip(top_logits.squeeze(0).tolist(), top_indices.squeeze(0).tolist()):
    print(f"{p:.2f}\t {i}\t {tokenizer.decode([i])}")

  # 최종 선택된 1순위 토큰 출력
  idx_next = torch.argmax(logits, dim=-1, keepdim=True)
  out = tokenizer.decode(idx_next.squeeze(0).tolist())
  print(f"Predicted next token: {idx_next.item()} -> '{out}'")

# 3. 실제 문장 생성 테스트 함수
def test2(model, tokenizer, device, keywords:str = "Harry Potter", filename:str = ""):
  """설정된 조건에 맞춰 모델이 이어지는 문장을 10번 생성하게 함"""
  idx = tokenizer.encode(keywords)
  idx = torch.tensor(idx).unsqueeze(0)

  # 모델의 위치 임베딩 가중치 크기로부터 context_size 파악
  context_size = model.pos_emb.weight.shape[0] 

  for i in range(10): # 10번 반복 생성 테스트
    token_ids = generate(
      model=model,
      idx=idx.to(device),
      max_new_tokens=50,
      context_size=context_size,
      top_k=50,       # 상위 50개 후보 중 선택
      temperature=0.5 # 적당한 창의성 부여
    )

    # 결과 출력 (줄바꿈 제거하여 가독성 확보)
    out = tokenizer.decode(token_ids.squeeze(0).tolist()).replace("\n", " ")
    print(i, ":", out)

# 4. 실행 진입점
def run():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = tiktoken.get_encoding("gpt2")
  keywords = "Malfoy is"
  
  torch.manual_seed(123)
  model = GPTModel()
  model.to(device)

  # model_dir에 저장된 첫 번째 가중치 파일을 불러와 테스트
  for filename in os.listdir(settings.model_dir):
    # 저장된 가중치(state_dict) 로드
    model.load_state_dict(torch.load(os.path.join(settings.model_dir, filename), 
                                      map_location=device, weights_only=True))
    model.eval() # 추론 모드로 전환
    
    test2(model, tokenizer, device, keywords, filename)
    break # 첫 번째 모델 파일만 테스트 후 종료
