import os
from settings import settings
import tiktoken
from src.train_model import GPTModel
from src.step5 import test2
import torch
from safetensors.torch import load_file

def run(last_model_name):
  # 1. 연산 장치 설정: GPU(cuda) 사용이 가능하면 GPU를, 아니면 CPU를 할당합니다.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # 2. Safetensors 가중치 로드: 
  # Pickle 기반의 .pth보다 보안이 뛰어나고 속도가 빠른 .safetensors 파일을 읽어옵니다.
  # settings.model_dir 경로와 전달받은 파일명을 합쳐 전체 경로를 구성합니다.
  weights = load_file(os.path.join(settings.model_dir, last_model_name))

  # 3. 모델 객체 생성: train_model.py에 정의된 GPT-2 구조의 빈 모델을 생성합니다.
  model = GPTModel()
  
  # 4. 가중치 주입: 로드한 weights(딕셔너리 형태)를 모델의 각 레이어에 복사합니다.
  model.load_state_dict(weights)
  
  # 5. 모델을 장치로 이동: 모델의 파라미터들을 설정된 GPU 또는 CPU 메모리로 보냅니다.
  model.to(device)
  
  # 6. 평가 모드 전환: 추론(Inference)을 위해 드롭아웃(Dropout) 등 학습용 기능을 비활성화합니다.
  model.eval()

  # 7. 토크나이저 준비: 텍스트를 모델이 이해하는 숫자(ID)로 바꾸기 위해 GPT-2 인코딩을 불러옵니다.
  tokenizer = tiktoken.get_encoding("gpt2")
  
  # 8. 입력 텍스트(프롬프트) 설정: 모델에게 줄 시작 문장입니다.
  keywords = "Shut up"

  # 9. 문장 생성 실행: 
  # src/step5.py의 test2 함수를 호출하여 키워드 뒤에 올 토큰들을 생성하고 출력합니다.
  # 마지막 인자 ""는 추가적인 설정값(예: 저장 경로 등)을 위해 비워둔 것으로 보입니다.
  test2(model, tokenizer, device, keywords, "")
