from unsloth import FastLanguageModel

def run():
  # 1. 사전 학습된 모델과 토크나이저를 불러옵니다.
  model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # 불러올 모델의 경로 또는 허브 이름
    max_seq_length = 2048, # 모델이 한 번에 처리할 최대 토큰 길이 (메모리에 영향)
    dtype = None,          # 데이터 타입 (None 설정 시 시스템에 맞춰 자동 감지: float16, bfloat16 등)
    load_in_4bit = True,   # 4비트 양자화(Quantization) 사용 여부. 메모리 사용량을 획기적으로 줄여줌
  )

  # 2. 효율적인 학습을 위한 PEFT(Parameter-Efficient Fine-Tuning) 설정을 적용합니다.
  # 주로 LoRA 기법을 사용하여 모델 전체가 아닌 일부 가중치만 학습하도록 만듭니다.
  model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                # LoRA Rank: 업데이트할 행렬의 크기. 높을수록 정교하지만 메모리를 더 사용함
    lora_alpha = 32,       # LoRA의 가중치 스케일링 계수. 보통 r의 2배 정도로 설정
    lora_dropout = 0.05,   # 드롭아웃 확률 (0으로 설정 시 최적화에 유리하지만, 과적합 방지를 위해 0.05 사용 가능)
    target_modules = [     # LoRA를 적용할 모델 내의 특정 레이어(모듈) 지정
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention 관련 모듈
        "gate_proj", "up_proj", "down_proj",      # MLP/Feed-forward 관련 모듈
    ],
    bias = "none",         # 바이어스 학습 여부. 일반적인 LoRA 학습에서는 "none" 권장
    # Unsloth의 최적화된 그레이디언트 체크포인팅 사용 (메모리 절약 기술)
    use_gradient_checkpointing = "unsloth",
    random_state = 123,    # 재현성을 위한 랜덤 시드 설정
    use_rslora = False,    # Rank Stabilized LoRA 사용 여부
    loftq_config = None,   # LoftQ(양자화 초기화 기술) 설정 여부
  )

  return model, tokenizer

if __name__ == "__main__":
  run()
