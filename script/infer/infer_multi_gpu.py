import argparse
import json
import math
import os
from peft import AutopeftModelForCausalLM, PeftModelForCausalLM
from PIL import Image
import torch
import torch.nn.functional as F
from torch.multiprocessing import Manager, Process, set_start_method
from torch.types import Device
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForcausalLM
from typing import List, Optional, Tuple

import sys
sys.path.append('./')

from src.model import LlamaForCausalLM

CAMERA_FRONT_TOKEN ="<camera>front</camera>"
CAMERA_FRONT_LEFT_TOKEN = "<camera>front left</camera>"
CAMERA_FRONT_RIGHT_TOKEN = "<camera>front right</camera>"
CAMERA_BACK_TOKEN = "<camera>back</camera>"
CAMERA_BACK_LEFT_TOKEN = "<camera>back left</camera>"
CAMERA_BACK_RIGHT_TOKEN = "<camera>back right</camera>"

BEV_TOKEN = "<bev>"

class CustomDataset(Dataset):
    def init (self, data):
        self.data = data
    
    def  len (self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        id = item['id']
        # task = item['task_key']
        if isinstance(item['image'], list):
            # item[' image'] = [os.path.join("/tmp/dataset/dataset-803-version-1/trainval/", path) for path in item['image']]
            image =[Image.open(img).convert('RGB') for img in item['image']]
            frame_id = id.split("")[0] + ".bin"
            bev_path = os.path.join("/tmp/dataset/dataset-803-version-1/sparsebev/vit without future/trainval/", frame_id)
            if os.path.isfile(bev_path):
                bev = torch.load(bev_path)
        else:
            image = Image.open(item['image']).convert('RGB' )
        conversations = item['conversations']
        return {'id': id, 'image': image, 'bev': bev, 'conversations': conversations}
    
def collate_fn(batch):
    ids = [item['id'] for item in batch]
    bevs = torch.stack([item['bev'] for item in batch])
    # tasks = [item['task'] for item in batch]
    images = [item['image'] for item in batch]
    prompts = [item['conversations'][0]['value'].replace('<image>\n', '') for item in batch]
    gts = [item['conversations'][1]['value'] for item in batch]
    return {'id': ids, 'images': images, 'bevs': bevs, 'prompts': prompts, 'gts': gts}

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

@torch.inference_mode()
def generate(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    device: Device,
    input_ids: List[List[int]],
    images: torch.Tensor,
    bevs: torch.Tensor,
    max_gen_len: int,
    num_image_tokens: int = 630,
    temperature: float = 0,
    top_p: float = 0.9,
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    assert images.ndim == 5
    bsz = len(input_ids)

    min_prompt_len = min(len(input_id) for input_id in input_ids)
    max_prompt_len = max(len(input_id) for input_id in input_ids)
    assert max_prompt_len + num_image_tokens <= model.config.tokenizer_model_max_length
    total_len = min(model.config.tokenizer_model_max_length - num_image_tokens, max_gen_len + num_image_tokens)

    pad_id = tokenizer.pad_token_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.full((bsz, total_len), 1, dtype=torch.long, device=device)

    for k, t in enumerate(input_ids):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device=device)
    input_text_mask = tokens != pad_id

    stop_token = torch.tensor(tokenizer.eos_token_id, device=device)
    past_key_values = None
    cache_position = torch.arange(0, total_len, device=device)
    for cur_pos in range(min_prompt_len, total_len):
        _input = model.prepare_inputs_for_generation(input_ids=tokens[:, 0: cur_pos],
                                                     past_key_values=past_key_values,
                                                     attention_mask=attention_mask[:, 0: cur_pos],
                                                     iamges=images,
                                                     use_cache=True,
                                                     cache_position=cache_position[prev_pos:cur_pos])
        out = model.forward(input_ids=_input['input_ids'],
                            attention_mask=_input['attention_mask'],
                            position_ids=_input['position_ids'],
                            past_key_values=_input['past_key_values'],
                            use_cache=_input['use_cache'],
                            output_attentions=False,
                            output_hidden_states=False,
                            images=_input['images'],
                            bevs=bevs,
                            return_dict=True,
                            cache_position=cache_position[prev_pos:cur_pos])
        logits = out.logits
        past_key_values = out.past_key_values
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token

        eos_reached |= (~input_text_mask[:, cur_pos]) & (torch.isin(next_token, stop_token))
        prev_pos = cur_pos
        if all(eos_reached):
            break

    out_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = len(input_ids[i])
        toks = toks[start: len(input_ids[i]) + max_gen_len]
        # cut to after eos tok if any
        try:
            eos_idx = toks.index(stop_token)
            toks = toks[:eos_idx]
        except ValueError:
            pass
        out_tokens.append(toks)
    return out_tokens

def inference(rank, gpu_id, args, data_dict):
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda")
    offset_bos = 1

    model = LlamaForCausalLM.from_pretrained(
        args.path_to_adapter,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.path_to_adapter,
        trust_remote_code=True,
        use_fast=True
    )

    # mm_projector_weight = torch.load(f"{args.path_to_adapter}/mm_projector.bin", map_location="cpu")
    # model.load_state_dict(mm_projector_weight, strict=False)

    # model_state_dict = model.state_dict()
    # for key in mm_projector_weight.keys():
    #     if key not in model_state_dict:
    #         raise KeyError(f"Key {key} is wrong in the weight.")
    #     if model_state_dict[key].shape != mm_projector_weight[key].shape:
    #         raise ValueError(f"shape missmatch for {key}: expected {model_state_dict[key].shape}, got {mm_projector_weight[key].shape}")

    model.to(device)
    model.eval()

    with open(args.eval_json_path, "rb") as file:
        data_all = json.load(file)

    num_processes = args.num_processes
    data_per_process = math.ceil(len(data_all) / num_processes)
    start_idx = rank * data_per_process
    end_idx = min((rank + 1) * data_per_process, len(data_all))
    data_to_process = data_all[start_idx: end_idx]
    dataset = CustomDataset(data_to_process)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            ids = batch['ids']
            images = batch['images']
            tasks = batch['tasks']
            bevs = batch['bevs'].to(dtype=model.dtype, device=device)
            prompts = batch['prompts']
            gts = batch['gts']

            input_ids = []
            for prompt in prompts:
                text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:<image>\n{prompt} ASSISTANT:"
                text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
                input_ids.append(text_chunks[0] + [-200] + text_chunks[1][offset_bos:])
                input_ids.append(
                    text_chunks[0] + 
                    tokenizer(CAMERA_FRONT_TOKEN).input_ids[offset_bos:] + [-201] +
                    tokenizer(CAMERA_FRONT_LEFT_TOKEN).input_ids[offset_bos:] + [-201] +
                    tokenizer(CAMERA_FRONT_RIGHT_TOKEN).input_ids[offset_bos:] + [-201] +
                    tokenizer(CAMERA_BACK_TOKEN).input_ids[offset_bos:] + [-201] +
                    tokenizer(CAMERA_BACK_LEFT_TOKEN).input_ids[offset_bos:] + [-201] +
                    tokenizer(CAMERA_BACK_RIGHT_TOKEN).input_ids[offset_bos:] + [-201] +
                    tokenizer(BEV_TOKEN).input_ids[offset_bos:] + [-207] +
                    text_chunks[1][offset_bos:]
                )
            if len(images[0]) > 1:
                image_tensor = model.process_multiple_images(images, model.config).to(dtype=model.dtype, device=device)
            else:
                image_tensor = model.process_images(batch['images'], model.config).to(dtype=model.dtype, device=device)
            
            output_ids = generate(model, tokenizer, device, input_ids, images=image_tensor, bevs=bevs, max_gen_len=512)
            results = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for i, result in enumerate(results):
                print(f"Pricess {rank}: Result - {result.strip()}")
                print(f"Process {rank}: GT - {gts[i]}")
                data_dict.append({'id': ids[i], 'question': prompts[i], 'answer': result.strip(), 'gt': gts[i]})

    print(f"Process {rank} finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('--path_to_adapter', type=str, default="")
    parser.add_argument('--eval_json_path', type=str, default="", help="path to eval file.")
    parser.add_argument('--output_json_path', type=str, default="", help="path to prediction file.")
    parser.add_argument('--num_processes', type=int, default=8, help="number of processes to execute.")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size for each gpu.")
    args = parser.parse_args()

    set_start_method("spawn")

    processes = []
    manager = Manager()
    data_dict = manager.list()
    for rank in range(args.num_processes):
        p = Process(target=inference, args=(rank, rank, args, data_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(args.output_json_path, "w") as f:
        json.dump(list(data_dict), f, indent=4)