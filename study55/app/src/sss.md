import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
from unsloth import FastLanguageModel
# 앞서 작성하신 파일을 model_loader.py라고 가정하고 run 함수를 가져옵니다.
# 파일명이 다르다면 해당 파일명으로 수정하세요.
from model_loader import run as load_model 

def run():
    # --- [STEP 1 & 2: 모델 로드 및 LoRA 설정] ---
    # 이전에 만드신 함수를 호출하여 준비된 모델과 토크나이저를 가져옵니다.
    print("--- 1. 모델 및 토크나이저 로딩 중 ---")
    model, tokenizer = load_model()

    # --- [STEP 3: 데이터셋 준비] ---
    print("--- 2. 데이터셋 준비 중 ---")
    # 예시: 'dataset.jsonl' 파일을 로드합니다. (본인의 파일명에 맞게 수정)
    # 데이터셋은 {"text": "질문과 답변 내용..."} 형식을 권장합니다.
    try:
        dataset = load_dataset("json", data_files="my_dataset.jsonl", split="train")
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return

    # --- [STEP 4: 학습 실행 (Training)] ---
    print("--- 3. 학습 시작 ---")
    tokenizer.padding_side = "right"
    max_seq_length = 2048

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = SFTConfig(
            output_dir = "outputs",
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none",
        ),
    )

    trainer_stats = trainer.train()
    print("--- 학습 완료 ---")

    # --- [STEP 5: 추론 테스트 (Inference)] ---
    print("--- 4. 추론 테스트 시작 ---")
    FastLanguageModel.for_inference(model) # 추론 모드 전환 (2배 빠름)

    class StopOnToken(StoppingCriteria):
        def __init__(self, stop_token_id):
            self.stop_token_id = stop_token_id
        def __call__(self, input_ids, scores, **kwargs):
            return self.stop_token_id in input_ids[0]

    stop_token = "<|end_of_text|>"
    stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)[0]
    stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])

    inputs = tokenizer(
        ["날씨 어때?"], # 테스트 질문
        return_tensors = "pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        streamer = text_streamer,
        max_new_tokens = 1024,
        stopping_criteria = stopping_criteria
    )

    # --- [STEP 6: 저장 및 허브 공유 (Push)] ---
    print("--- 5. 모델 저장 및 GGUF 업로드 시작 ---")
    
    # 설정값 (본인의 환경에 맞게 수정 필수)
    save_directory = "20260406_model"
    huggingface_repo = "YourID/YourModelName" # 예: "gemma-2-9b-it-korean"
    huggingface_token = "hf_your_token_here" # HuggingFace Write 토큰
    quantization_method = "q4_k_m" # GGUF 양자화 방식

    # 로컬 저장
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    # 허브 업로드 (GGUF 변환 포함)
    model.push_to_hub_gguf(
        huggingface_repo + "-gguf",
        tokenizer,
        quantization_method = quantization_method,
        token = huggingface_token,
    )
    print(f"--- 모든 과정이 완료되었습니다. 허브 주소: https://huggingface.co/{huggingface_repo}-gguf ---")

if __name__ == "__main__":
    run()