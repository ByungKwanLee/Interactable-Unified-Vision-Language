from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers import AutoTokenizer

from .transformer import *
from .build import *
from modeling.language.LangEncoder import conversation as conversation_lib


def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    model_name = config_encoder['NAME']

    if not is_lang_encoder(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return lang_encoders(model_name)(config_encoder, tokenizer, verbose, **kwargs)

# def build_tokenizer(config_encoder):
#     tokenizer = None
#     os.environ['TOKENIZERS_PARALLELISM'] = 'true'
#     if config_encoder['TOKENIZER'] == 'clip':
#         pretrained_tokenizer = config_encoder.get(
#             'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
#         )
#         tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
#         tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
#     elif config_encoder['TOKENIZER'] == 'clip-fast':
#         pretrained_tokenizer = config_encoder.get(
#             'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
#         )
#         tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_tokenizer, from_slow=True)
#     else:
#         pretrained_tokenizer = config_encoder.get(
#             'PRETRAINED_TOKENIZER', 'lmsys/vicuna-7b-v1.5'
#         )
#         tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer, model_max_length=256,
#             padding_side="right", use_fast=False,)
#         tokenizer.pad_token = tokenizer.unk_token
        
#         conversation_lib.default_conversation = conversation_lib.conv_templates['v1']
        
#     return tokenizer

def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if config_encoder['TOKENIZER'] == 'clip':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        # tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer, model_max_length=256,
            padding_side="right", use_fast=False,)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates['v1']

        

    elif config_encoder['TOKENIZER'] == 'clip-fast':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_tokenizer, from_slow=True)
    else:
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'lmsys/vicuna-7b-v1.5'
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer, model_max_length=256,
            padding_side="right", use_fast=False,)
        tokenizer.pad_token = tokenizer.unk_token
        
        conversation_lib.default_conversation = conversation_lib.conv_templates['v1']
        
    return tokenizer