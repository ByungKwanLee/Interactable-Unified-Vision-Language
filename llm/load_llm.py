import torch
import pathlib
from transformers import BitsAndBytesConfig

from .llava import LlavaLlamaForCausalLM
from .utils import *

class Argument:
    bf16 = True
    double_quant = True
    quant_type = 'nf4' #fp4
    bits = 4

def prepare_llm(ckpt="/mnt/hard1/lbk-cvpr/checkpoints/vicuna-7b-v1.3"):
    args = Argument()
    
    bnb_model_from_pretrained_args = {}
    if args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type # {'fp4', 'nf4'}
            )
        ))

    model = LlavaLlamaForCausalLM.from_pretrained(ckpt, cache_dir=False, **bnb_model_from_pretrained_args)
    if args.bits == 16 and args.bf16: model.to(torch.bfloat16)
            
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        ckpt,
        cache_dir=False,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token

    # penetrate return
    return model, tokenizer, DataCollatorForSupervisedDataset(tokenizer=tokenizer)

