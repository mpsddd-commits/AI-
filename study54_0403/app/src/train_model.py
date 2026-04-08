import tiktoken
import torch
import torch.nn as nn

# --- 하이퍼파라미터 설정 ---
tokenizer = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = tokenizer.n_vocab      # 어휘 사전 크기 (GPT-2 기준 50,257)
CONTEXT_LENGTH = 128                # 한 번에 처리할 최대 시퀀스 길이
EMB_DIM = 768                       # 임베딩 벡터의 차원
NUM_HEADS = 12                      # 멀티-헤드 어텐션의 헤드 개수
NUM_LAYERS = 12                     # 트랜스포머 블록 쌓는 횟수
DROP_RATE = 0.1                     # 드롭아웃 확률
QKV_BIAS = False                    # Query, Key, Value 선형 층의 편향 사용 여부

class MultiHeadAttention(nn.Module):
  """
  입력을 여러 개의 헤드로 나누어 병렬로 어텐션을 수행하는 클래스
  """
  def __init__(self, d_in, d_out):
    super().__init__()
    assert d_out % NUM_HEADS == 0, "d_out은 NUM_HEADS로 나누어 떨어져야 합니다."

    self.d_out = d_out
    self.head_dim = d_out // NUM_HEADS # 각 헤드의 차원

    # Query, Key, Value를 얻기 위한 선형 변환
    self.W_query = nn.Linear(d_in, d_out, bias=QKV_BIAS)
    self.W_key = nn.Linear(d_in, d_out, bias=QKV_BIAS)
    self.W_value = nn.Linear(d_in, d_out, bias=QKV_BIAS)
    
    # 멀티헤드 결과를 합친 후 최종 출력을 위한 투영
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(DROP_RATE)
    
    # 인과적 마스크(Causal Mask): 미래의 토큰을 보지 못하게 차단
    # 상삼각 행렬(triu)을 생성하여 자기 회귀적 특성을 유지함
    self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1))

  def forward(self, x):
    b, num_tokens, d_in = x.shape

    # 1. 선형 변환 수행
    keys = self.W_key(x)    # (b, num_tokens, d_out)
    queries = self.W_query(x)
    values = self.W_value(x)

    # 2. 헤드 단위로 텐서 모양 변경 (멀티헤드 분할)
    # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
    keys = keys.view(b, num_tokens, NUM_HEADS, self.head_dim)
    values = values.view(b, num_tokens, NUM_HEADS, self.head_dim)
    queries = queries.view(b, num_tokens, NUM_HEADS, self.head_dim)

    # 3. 차원 맞교환: (b, num_heads, num_tokens, head_dim)
    # 행렬 곱을 위해 헤드 차원을 앞으로 보냄
    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)

    # 4. 점곱 어텐션 점수 계산: (b, num_heads, num_tokens, num_tokens)
    attn_scores = queries @ keys.transpose(2, 3)

    # 5. 마스킹 적용: 미래 토큰 위치에 -무한대를 채워 softmax 시 0이 되게 함
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    attn_scores.masked_fill_(mask_bool, -torch.inf)

    # 6. 소프트맥스 및 스케일링
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    # 7. 가중치와 Value 결합: (b, num_heads, num_tokens, head_dim)
    context_vec = (attn_weights @ values).transpose(1, 2)

    # 8. 모든 헤드 다시 합치기: (b, num_tokens, d_out)
    context_vec = context_vec.reshape(b, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec)

    return context_vec

class LayerNorm(nn.Module):
  """
  입력의 마지막 차원을 기준으로 평균과 표준편차를 구해 정규화하는 클래스
  """
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim)) # 학습 가능한 가중치 (Gamma)
    self.shift = nn.Parameter(torch.zeros(emb_dim)) # 학습 가능한 편향 (Beta)

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift

class GELU(nn.Module):
  """
  GPT-2에서 사용하는 활성화 함수 (Gaussian Error Linear Unit)
  Relu보다 부드러운 곡선을 가짐
  """
  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
      torch.sqrt(torch.tensor(2.0 / torch.pi)) *
      (x + 0.044715 * torch.pow(x, 3))
    ))

class FeedForward(nn.Module):
  """
  어텐션 결과에 비선형성을 추가하는 층
  일반적으로 임베딩 차원의 4배로 확장했다가 다시 축소함
  """
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(EMB_DIM, 4 * EMB_DIM),
      GELU(),
      nn.Linear(4 * EMB_DIM, EMB_DIM),
    )

  def forward(self, x):
    return self.layers(x)

class TransformerBlock(nn.Module):
  """
  GPT 모델의 기본 구성 단위
  Attention -> Add & Norm -> FeedForward -> Add & Norm 구조
  """
  def __init__(self):
    super().__init__()
    self.att = MultiHeadAttention(d_in=EMB_DIM, d_out=EMB_DIM)
    self.ff = FeedForward()
    self.norm1 = LayerNorm(EMB_DIM)
    self.norm2 = LayerNorm(EMB_DIM)
    self.drop_shortcut = nn.Dropout(DROP_RATE)

  def forward(self, x):
    # 1. 어텐션 경로 (잔차 연결 포함)
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + shortcut # 입력값과 처리값을 더함

    # 2. 피드포워드 경로 (잔차 연결 포함)
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut

    return x

class GPTModel(nn.Module):
  """
  최종 GPT 모델 클래스
  """
  def __init__(self):
    super().__init__()
    # 토큰 임베딩: 단어 ID를 벡터로 변환
    self.tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
    # 위치 임베딩: 토큰의 위치 정보를 벡터로 추가
    self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM)
    self.drop_emb = nn.Dropout(DROP_RATE)

    # 여러 개의 트랜스포머 블록을 순차적으로 쌓음
    self.trf_blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])

    # 최종 레이어 정규화 및 출력 헤드(어휘 사전 크기로 투영)
    self.final_norm = LayerNorm(EMB_DIM)
    self.out_head = nn.Linear(EMB_DIM, VOCAB_SIZE, bias=False)

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape
    
    # 입력된 토큰 인덱스들에 위치 정보를 더함
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
    
    x = tok_embeds + pos_embeds 
    x = self.drop_emb(x)
    
    # 트랜스포머 블록 통과
    x = self.trf_blocks(x)
    
    # 정규화 후 로짓(Logits) 생성
    x = self.final_norm(x)
    logits = self.out_head(x)
    
    return logits