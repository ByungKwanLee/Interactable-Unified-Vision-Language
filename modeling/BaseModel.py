import os
import logging

import torch
import torch.nn as nn

from utils.model import align_and_update_state_dicts

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        return outputs

    def save_pretrained(self, save_dir, epoch):
        model_state_dict = self.model.state_dict()
        filtered_model_keys = list(filter(lambda x: not x.startswith('llm.'), model_state_dict))
        result_dicts = {}
        for model_key in filtered_model_keys:
            result_dicts[model_key] = model_state_dict[model_key]
        os.makedirs(os.path.join(save_dir, f'epoch{epoch}'), exist_ok=True)
        torch.save(result_dicts, os.path.join(save_dir, f'epoch{epoch}', f"syslearner.pt"))
        if self.opt['Load_LLM']:
            llm_path = os.path.join(save_dir, f'epoch{epoch}', "llm")
            self.model.llm.save_pretrained(llm_path)
            self.model.llm_tokenizer.save_pretrained(llm_path)

    def from_pretrained(self, load_dir):
        if self.opt['Load_LLM']:
            try:
                self.model.llm.from_pretrained(os.path.join("/".join(load_dir.split('/')[:-1]), 'llm'))
                self.model.llm_tokenizer.from_pretrained(os.path.join("/".join(load_dir.split('/')[:-1]), 'llm'))
            except:
                print('There are no LLM pretrained file: {}'.format(os.path.join("/".join(load_dir.split('/')[:-1]))))
        state_dict = torch.load(load_dir, map_location=self.opt['device'])
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self