import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from model.dspec_model import DSPModel
from fastchat.model import get_conversation_template
import torch
model = DSPModel.from_pretrained(
    base_model_path="/share/public/public_models/Qwen2.5-7B-Instruct",
    ea_model_path="/share/public/wanghanzhen/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()
prompt="Hello, how to play basketball? "
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.dsp_generate(input_ids,temperature=0.5,max_new_tokens=512)
output=model.tokenizer.decode(output_ids[0])
print(output)