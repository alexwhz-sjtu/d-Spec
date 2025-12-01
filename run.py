import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from model.dspec_model import DSPModel
from fastchat.model import get_conversation_template
import torch
model = DSPModel.from_pretrained(
    base_model_path="/share/public/public_models/Qwen3-8B",
    # base_model_path="/share/public/public_models/Qwen2.5-7B-Instruct",
    ea_model_path="/share/public/wanghanzhen/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()
# prompt='''小明有若干颗糖果。
# 他先把糖果的一半又一颗分给了小红；
# 然后把剩下的一半又一颗分给了小刚；
# 最后把剩下的糖果的一半又一颗分给了小丽；
# # 结果自己还剩1颗糖果。

# 问：小明最开始有多少颗糖果？ '''
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = model.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
print(f"input:{text}")
input_ids = model.tokenizer([text]).input_ids
# input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.dsp_generate(input_ids,temperature=0.5,max_length=4096)
output=model.tokenizer.decode(output_ids[0])
print(output)