import torch
import pathlib
from transformers import BitsAndBytesConfig

from .llava import LlavaLlamaForCausalLM
from .utils import *


def prepare_model_for_kbit_training(model):
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    # freeze 
    for param in model.parameters(): param.requires_grad = False

    # cast all non INT8 parameters to fp32
    # for param in model.parameters():
    #     if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
    #         param.data = param.data.to(torch.float32)

    if loaded_in_kbit:
        # For backward compatibility
        model.enable_input_require_grads()
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        
    return model


def prepare_llm(bits=16, double_quant=True, quant_type='nf4', ckpt="/mnt/ssd/lbk-cvpr/checkpoints/vicuna-7b-v1.5"):
    
    bnb_model_from_pretrained_args = {}
    if bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_quant_type=quant_type # {'fp4', 'nf4'}
            )
        ))
    model = LlavaLlamaForCausalLM.from_pretrained(ckpt, cache_dir=False, low_cpu_mem_usage=True, **bnb_model_from_pretrained_args)
    model.config.use_cache = False
    
    # PEFT for gradient checkpointing   
    if bits in [4, 8]: model.config.torch_dtype=torch.bfloat16
    model = prepare_model_for_kbit_training(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        ckpt,
        cache_dir=False,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float16)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    module = module.to(torch.bfloat16)

    """
    # [Step1] tokenizer 
    input_ids = tokenizer(
                "What the fucking LLM! Who are you?",
                return_tensors="pt",
                padding="longest",
                max_length=50,
                truncation=True,
            ).input_ids.cuda()
    
    
    # [Step2] llm prediction
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            min_length=1,
            no_repeat_ngram_size=3,
            num_beams=5)
        
    # [Step3] llm prediction to string
    llm_outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    llm_outputs = llm_outputs.strip()
    """

    # penetrate return
    return model, tokenizer, DataCollatorForSupervisedDataset(tokenizer=tokenizer)

