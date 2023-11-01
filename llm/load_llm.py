import torch
import pathlib
from .llava import LlavaLlamaForCausalLM
from .llava_trainer import LLaVATrainer
from transformers.trainer_utils import IntervalStrategy, SchedulerType


from .utils import *

local_rank = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/hard1/lbk-cvpr/checkpoints/vicuna-7b-v1.3")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=1)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    # json file -> code 
    bf16: bool = field(default=True)
    output_dir: str = field(default='/mnt/hard1/lbk-cvpr/checkpoints/llava-v1.3-7b-trial')
    num_train_epochs: float = field(default=1.0)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    evaluation_strategy: Union[IntervalStrategy, str] = field(default="no")
    save_strategy: Union[IntervalStrategy, str] = field(default="steps")
    save_steps: float = field(default=24000)
    save_total_limit: Optional[int] = field(default=1)

    learning_rate: float = field(default=2e-3)
    weight_decay: float = field(default=0.0)
    lr_scheduler_type: Union[SchedulerType, str] = field(default="cosine")
    logging_steps: float = field(default=1)
    tf32: Optional[bool] = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)

def prepare_llm():
    parser = BK_HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token

    # penetrate return
    return model, tokenizer 

