import streamlit as st
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# PEFT LORA 모델 및 설정 로드
peft_model_id = "AhmedSSoliman/Llama2-CodeGen-PEFT-QLoRA"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(peft_model_id, return_dict=True)

# tokenizer 및 model의 device 설정
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
device = torch.device("cuda")
model.to(device)

# 챗봇 생성 함수
def generate_response(instruction):
    inputs = tokenizer.encode(instruction, return_tensors="pt").to(device)
    
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )
    with torch.no_grad():
        generation_output = model.generate(
            inputs,
            max_length=128,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=generation_config,
        )

    generated_response = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    return generated_response

# Streamlit 애플리케이션 정의
st.title("챗봇")

# 사용자 입력 받기
instruction = st.text_area("입력:", "")

# 입력이 있을 때만 챗봇 응답 생성 및 출력
if instruction:
    response = generate_response(instruction)
    st.write("응답:", response)

