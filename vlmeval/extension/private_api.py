import json
import base64
from qcloud_cos import CosConfig, CosS3Client
from venus_api_base.config import Config
from venus_api_base.http_client import HttpClient

from ..api.base import BaseAPI
from ..smp.vlm import encode_image_to_base64


class TencentGPT:
    domain = 'http://v2.open.venus.oa.com'
    header = {
        'Content-Type': 'application/json',
    }

    def __init__(self, model='gpt-4-vision-preview', secret_id=None, secret_key=None, app_group_id=None):
        self.model = model
        self.client = HttpClient(config=Config(read_timeout=3000), secret_id=secret_id, secret_key=secret_key)
        self.app_group_id = int(app_group_id)

    def generate(self, messages, max_tokens=1024, temperature=0., **kwargs):
        assert isinstance(messages, list)

        body = {
            "appGroupId": self.app_group_id,
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        body.update(**kwargs)

        try:
            ret = self.client.post(f'{self.domain}/chat/single',
                                   header=self.header,
                                   body=json.dumps(body))
            if ret['code'] != 0:
                raise RuntimeError(f"Error1 in TencentGPT: {ret['traceId']}; {ret['message']}")
            elif ret['data']['status'] != 2:
                raise RuntimeError(f"Error2 in TencentGPT: {ret['traceId']}; {ret['data']['response']}")
            else:
                return ret['code'], ret['data']['response'], ret
        except Exception as e:
            raise RuntimeError(f"Error0 in TencentGPT: {e}")

    def chat(self, message, image_path=None, max_tokens=1024, temperature=0., **kwargs):
        messages = [{"role": "user",
                     "content": [{"type": "text",
                                  "text": message}]}]

        if image_path:
            with open(image_path, "rb") as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode('utf8')
            messages[-1]['content'].append({"type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})

        ret_code, answer, response = self.generate(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)

        return answer


class PrivateAPI(BaseAPI):
    def __init__(self, *args,
                 key: str = 'sk-oZ5RhVMixFUTDshEys5VbmIA/to1pEEfiwuWkv5SOAXYvYBVO/2010',
                 img_size: int = 512,
                 img_detail: str = 'low',
                 **kwargs):
        key = key[len('sk-'):]
        secret_id, secret_key, app_group_id = key.split('/')
        self.model = TencentGPT('gpt-4-turbo-2024-04-09', secret_id, secret_key, app_group_id)

        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail

        super().__init__(*args, **kwargs)

    def generate_inner(self, inputs, max_tokens=1024, temperature=0., **kwargs):
        inputs = self.prepare_inputs(inputs)
        ret_code, answer, response = self.model.generate(inputs, max_tokens=max_tokens, temperature=temperature)

        return ret_code, answer, response

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert all('type' in x for x in inputs) or all('role' in x for x in inputs), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def prepare_itlist(self, inputs):
        assert all(isinstance(x, dict) for x in inputs)
        has_images = any(x['type'] == 'image' for x in inputs)
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list