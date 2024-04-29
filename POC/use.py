"""
Pour utiliser le modele preentrainé dans le POC
"""

import torch, transformers, pyreft
device = "cuda"

# prompt_no_input_template = """\n<|user|>:%s</s>\n<|assistant|>:"""

prompt_no_input_template = """\n<|user|>:%s</s>\n<|assistant|>:"""

model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

reft_model = pyreft.ReftModel.load(
    "./POC/reft_to_share", model
)

reft_model.set_device("cuda") # send back to cpu before saving.

# get tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, 
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

#Use test
instruction = "Which dog breed do people think is cuter, poodle or doodle?"
while True:
    # tokenize and prepare the input
    prompt = prompt_no_input_template % instruction
    prompt = tokenizer(prompt, return_tensors="pt").to(device)

    base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
    _, reft_response = reft_model.generate(
        prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
        intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
        eos_token_id=tokenizer.eos_token_id, early_stopping=True
    )
    print(tokenizer.decode(reft_response[0], skip_special_tokens=True))

    instruction = input("\nPrompt: ")
