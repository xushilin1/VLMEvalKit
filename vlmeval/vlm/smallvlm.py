import torch
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen
from PIL import Image

import os
import math


class SmallVLM(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                model_path="shilinxu/SmallVLM",
                max_new_tokens=128,
                top_p=0.001,
                top_k=1,
                temperature=0.01,
                repetition_penalty=1.0,
                **kwargs):
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")

        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content
    

    def extract_vision_info(self, conversations: list[dict] | list[list[dict]]) -> list[dict]:
        vision_infos = []
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)
        return vision_infos

    def generate_inner(self, message, dataset=None):

        messages = []
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        
        image_inputs = []
        vision_infos = self.extract_vision_info(messages)
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                if isinstance(vision_info['image'], str):
                    image=Image.open(vision_info['image'])
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                image_inputs.append(image)
            elif 'image_url' in vision_info:
                image_inputs.append(vision_info['image_url'])
            else:
                raise ValueError("image, image_url or video should in content.")

        inputs = self.processor(text=text, images=image_inputs, padding=False, return_tensors='pt')
        inputs = inputs.to('cuda')
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        
        generated_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        return response