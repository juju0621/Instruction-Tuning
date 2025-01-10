from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是一名精通中文的教授，以下是用戶和人工智能助理之間的對話。 \
             你要對用戶的問題提供有用、安全、詳細和禮貌的回答。請將文言文翻譯成白話文或白話文翻譯成文言文。\
             這邊提供你兩個範例。\
             USER:翻譯成文言文：雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。答案：\
             ASSISTANT:雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。\
             USER:沒過十天，鮑泉果然被拘捕。幫我把這句話翻譯成文言文\
             ASSISTANT:後未旬，果見囚執。\
             USER: {instruction} ASSISTANT:"


def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
