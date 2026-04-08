import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
from unsloth import FastLanguageModel
from settings import settings
import step5 

def run():
    # --- [1. 모델 불러오기 및 PEFT 설정] ---
    print("--- Step 5에서 준비된 모델을 불러오는 중 ---")
    model, tokenizer = step5.run()

    # --- [2. 데이터셋 로드] ---
    # Step 2에서 업로드했던 본인의 허브 데이터셋을 다시 불러옵니다.
    print(f"--- 데이터셋 로딩 중: {settings.repo_name} ---")
    dataset = load_dataset(settings.repo_name, split="train")

    ## --- [전처리 추가: Alpaca 포맷팅] ---
    EOS_TOKEN = tokenizer.eos_token
    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Response:
    {}
    """

    def formatting_prompts_func(examples):
        instructions = examples["prompt"]     # Step 1에서 만든 prompt 키
        outputs = examples["completion"]      # Step 1에서 만든 completion 키
        texts = []
        for instruction, output in zip(instructions, outputs):
            # 질문과 답변을 템플릿에 넣고 끝에 EOS 토큰을 붙입니다.
            text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
            texts.append(text)
        return { "text": texts } # SFTTrainer가 인식할 'text' 필드 생성

    print("--- 데이터셋 포맷팅 중 ---")
    dataset = dataset.map(
        formatting_prompts_func,
        batched = True,
    )

    # --- [3. 학습 시작] ---
    print("--- 학습 시작 ---")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text", # 데이터셋 내의 텍스트 필드명 (Step 1 형식에 맞춤)
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False,
        args = SFTConfig(
            output_dir = "outputs",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,                # 짧은 학습 테스트용
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

    trainer.train()
    print("--- 학습 완료 ---")

    # --- [4. 추론 테스트 (Inference)] ---
    print("--- 학습된 모델 테스트 중 ---")
    FastLanguageModel.for_inference(model)

    # 중단 조건 설정
    class StopOnToken(StoppingCriteria):
        def __init__(self, stop_token_id):
            self.stop_token_id = stop_token_id
        def __call__(self, input_ids, scores, **kwargs):
            return self.stop_token_id in input_ids[0]

    stop_token_id = tokenizer.eos_token_id
    stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])

    # Step 1의 데이터 형식("prompt")에 맞춰 질문을 던집니다.
    inputs = tokenizer(["날씨 어때?"], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)

    _ = model.generate(
        **inputs,
        streamer = text_streamer,
        max_new_tokens = 128,
        stopping_criteria = stopping_criteria
    )

    # --- [5. 최종 저장 및 GGUF 허브 업로드] ---
    print("--- GGUF 변환 및 허브 업로드 시작 ---")
    
    # settings 파일에 정의된 정보를 활용합니다.
    # 보통 GGUF용 저장소는 별도로 관리하므로 이름 뒤에 -gguf를 붙입니다.
    gguf_repo = f"{settings.repo_name}-gguf"
    
    model.push_to_hub_gguf(
        repo_id = gguf_repo,
        tokenizer = tokenizer,
        quantization_method = "q4_k_m", # 표준적인 4비트 양자화
        token = settings.hf_token
    )
    print(f"--- 모든 과정 종료! 허브에서 '{gguf_repo}'를 확인하세요. ---")

if __name__ == "__main__":
    run()