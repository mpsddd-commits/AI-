import torch  # PyTorch 라이브러리를 불러옵니다. (딥러닝 프레임워크)

def run():
  # 1. 현재 설치된 PyTorch의 버전을 출력합니다.
  # 예: '2.6.0+cu124' (버전 번호와 함께 설치된 CUDA 버전이 표시될 수 있음)
  print(torch.__version__)

  # 2. 현재 사용 가능한 NVIDIA GPU의 '연산 능력(Compute Capability)' 버전을 가져옵니다.
  # major_version: 아키텍처의 큰 세대를 의미 (예: 8은 Ampere, 9는 Hopper 아키텍처)
  # minor_version: 해당 세대 내에서의 세부 개선 버전을 의미
  # 이 정보는 특정 라이브러리(예: FlashAttention)나 최적화 기법의 지원 여부를 판단할 때 필수적입니다.
  major_version, minor_version = torch.cuda.get_device_capability()

  # 3. 위에서 가져온 메이저 버전과 마이너 버전을 출력합니다.
  # 예: NVIDIA RTX 3090의 경우 (8, 6)이 출력됩니다.
  print(major_version, minor_version)

# 함수를 실행하려면 아래와 같이 호출해야 합니다.
if __name__ == "__main__":
  run()
