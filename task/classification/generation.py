# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # This prevents tokenizers from taking all cpus
import sys
import random
import pickle
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import pandas as pd
# Pytorch Modules
import torch
torch.set_num_threads(2) # This prevents Pytorch from taking all cpus
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# HuggingFace Modules
from transformers import AutoModelForCausalLM, AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.dataset import GenerationDataset, INSTRUCTION_MODELS
from utils.utils import TqdmLoggingHandler, write_log, check_path, get_torch_device, get_huggingface_model_name

def get_gen_file_suffix(args):
    gen_model = args.gen_model_type if args.gen_model_type else args.model_type
    return f'{gen_model}_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}'

def generation(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)

    # Define logger and tensorboard writer
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Load dataset and define dataloader - generalized for any number of labels
    dataset_gen, dataloader_gen = {}, {}
    # Create a dataset for each label
    first_ds = GenerationDataset(args, label=0)
    label_list = first_ds.label_list
    num_labels = len(label_list)
    dataset_gen[0] = first_ds
    for label_idx in range(1, num_labels):
        dataset_gen[label_idx] = GenerationDataset(args, label=label_idx)
    for label_idx in range(num_labels):
        dataloader_gen[label_idx] = DataLoader(dataset_gen[label_idx], batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True, drop_last=False)

    is_instruction_model = args.model_type in INSTRUCTION_MODELS
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Number of labels: {num_labels}, Label list: {label_list}")

    # Get model instance
    write_log(logger, "Building model")
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = dataset_gen[0].tokenizer

    if is_instruction_model:
        if args.use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    elif args.model_type in ['gpt2', 'gpt2_large', 'gpt2_xl', 'opt', 'bloom']:
        if args.use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    else:
        raise NotImplementedError(f"Model type {args.model_type} is not implemented.")

    write_log(logger, f"Model loaded: {model_name} (instruction={is_instruction_model}, 4bit={args.use_4bit})")

    # Start generation
    model = model.eval()
    generated_data = []
    per_label_amount = args.gen_amount // num_labels

    for current_label_idx in range(num_labels):
        label_name = label_list[current_label_idx]
        label_generated = []
        write_log(logger, f"Generating for label {current_label_idx}: '{label_name}' (target: {per_label_amount})")

        while len(label_generated) < per_label_amount:
            for iter_idx, data_dicts in enumerate(tqdm(dataloader_gen[current_label_idx],
                                                       total=len(dataloader_gen[current_label_idx]),
                                                       desc=f"Generation-{label_name.upper()}", position=0, leave=True)):
                if len(label_generated) >= per_label_amount:
                    break

                # Gen - Get input data
                labels = data_dicts['label']
                label_idx_batch = data_dicts['label_idx']

                # Gen - STAGE 1: Generate text
                input_prompt = []
                for label in labels:
                    if is_instruction_model:
                        input_prompt.append(dataset_gen[current_label_idx].build_chat_prompt(stage=1, label=label))
                    else:
                        input_prompt.append(dataset_gen[current_label_idx].build_prompt(stage=1, label=label))
                input_tokenized = tokenizer(input_prompt, return_tensors='pt', padding=True).to(device)
                prompt_length = input_tokenized['input_ids'].shape[1]

                with torch.no_grad():
                    outputs = model.generate(
                        input_tokenized['input_ids'],
                        attention_mask=input_tokenized['attention_mask'],
                        max_new_tokens=args.max_seq_len,
                        top_k=args.gen_top_k,
                        top_p=args.gen_top_p,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=args.gen_temperature,
                    )

                if is_instruction_model:
                    # For instruction models, only decode the generated part (after prompt)
                    generated_ids = outputs[:, prompt_length:]
                    decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    outputs_stage1 = [process_output_instruction(o) for o in decoded_outputs]
                else:
                    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    outputs_stage1 = [process_output(o) for o in decoded_outputs]

                # Gen - STAGE 2: Pseudo-labeling to filter out irrelevant sentences
                if args.gen_relabel != 'none':
                    input_prompt = []
                    for each_label in label_list:
                        for each_output in outputs_stage1:
                            if is_instruction_model:
                                input_prompt.append(dataset_gen[current_label_idx].build_chat_prompt(stage=2, input_text=each_output, label=each_label))
                            else:
                                input_prompt.append(dataset_gen[current_label_idx].build_prompt(stage=2, input_text=each_output, label=each_label))
                    input_tokenized = tokenizer(input_prompt, return_tensors='pt', padding=True).to(device)

                    # Divide into sub-batches and compute loss incrementally to save VRAM
                    total_samples = input_tokenized['input_ids'].size(0)
                    avg_loss_all = []
                    relabel_subbatch_size = max(1, args.batch_size // 4)  # Use smaller sub-batch for Stage 2
                    for i in range(0, total_samples, relabel_subbatch_size):
                        end_idx = min(i + relabel_subbatch_size, total_samples)
                        subbatch = {k: v[i:end_idx] for k, v in input_tokenized.items()}
                        with torch.no_grad():
                            output = model(**subbatch)
                            shifted_logits = output.logits[:, :-1, :].contiguous()
                            shifted_labels = subbatch['input_ids'][:, 1:].contiguous()
                            loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                           shifted_labels.view(-1)).view(shifted_labels.size())
                            avg_loss = loss.sum(-1) / (loss > 0).sum(-1).float()
                            avg_loss_all.append(avg_loss.cpu())
                            del output, shifted_logits, shifted_labels, loss, avg_loss
                    del input_tokenized
                    torch.cuda.empty_cache()

                    if len(avg_loss_all) == 0:
                        continue

                    # Reshape: total_samples = num_labels * batch_actual_size
                    # avg_loss_all is flat; reshape to (num_labels, batch_actual_size) then transpose
                    all_losses = torch.cat(avg_loss_all, dim=0)  # (num_labels * batch_actual,)
                    batch_actual = len(outputs_stage1)
                    pred = all_losses.view(num_labels, batch_actual).t().to(device)  # (batch_actual, num_labels)
                    prob = torch.softmax(pred / args.gen_relabel_temperature, dim=-1)
                    # Small loss -> More appropriate sentence -> reverse prob
                    prob = 1 - prob

                    # Filter: keep sentences with high probability exceeding threshold and matching current label
                    prob_max = prob.max(dim=-1)[0]
                    prob_max_exceed = prob_max > (1 / num_labels) + args.gen_relabel_threshold
                    prob_max_match = prob.argmax(dim=-1) == torch.tensor(current_label_idx).to(device)

                    outputs_stage2 = []
                    for i in range(len(prob_max_exceed)):
                        if prob_max_exceed[i] and prob_max_match[i] and len(outputs_stage1[i].strip()) > 0:
                            outputs_stage2.append({
                                'sentence': outputs_stage1[i],
                                'label_soft': [round(prob[i][j].item(), 2) for j in range(len(prob[i]))],
                                'label_hard': prob[i].argmax().item(),
                                'label_noisy': label_idx_batch[i].item(),
                            })
                else:
                    outputs_stage2 = []
                    for i in range(len(outputs_stage1)):
                        if len(outputs_stage1[i].strip()) > 0:
                            outputs_stage2.append({
                                'sentence': outputs_stage1[i],
                                'label_soft': [0.0 for j in range(num_labels)],
                                'label_hard': -1,
                                'label_noisy': label_idx_batch[i].item(),
                            })

                if len(outputs_stage2) > 0:
                    label_generated.extend(outputs_stage2)

        # Trim to exact amount
        if len(label_generated) > per_label_amount:
            label_generated = label_generated[:per_label_amount]
        generated_data.extend(label_generated)
        write_log(logger, f"Label '{label_name}': generated {len(label_generated)} samples")

    # Trim total to gen_amount
    if len(generated_data) > args.gen_amount:
        generated_data = generated_data[:args.gen_amount]

    write_log(logger, f"Total generated: {len(generated_data)} samples")

    # Transform generated data into data_dict format
    data_dict = {}
    for prefix in ['train', 'valid']:
        for suffix in ['NL', 'SL', 'HL']:
            data_dict[f'{prefix}_{suffix}'] = {
                'input_text': [],
                'labels': [],
                'soft_labels': [],
                'num_classes': num_labels,
            }

    # Shuffle generated data
    random.shuffle(generated_data)

    # Split generated data into train and valid
    valid_amount = int(len(generated_data) * args.train_valid_split) # 0.1
    train_data = generated_data[valid_amount:]
    valid_data = generated_data[:valid_amount]

    # Save each data into data_dict
    for split_name, split_data in [('train', train_data), ('valid', valid_data)]:
        for i in range(len(split_data)):
            data_dict[f'{split_name}_NL']['input_text'].append(split_data[i]['sentence'])
            data_dict[f'{split_name}_HL']['input_text'].append(split_data[i]['sentence'])
            data_dict[f'{split_name}_SL']['input_text'].append(split_data[i]['sentence'])

            data_dict[f'{split_name}_NL']['labels'].append(split_data[i]['label_noisy'])
            data_dict[f'{split_name}_HL']['labels'].append(split_data[i]['label_hard'])
            data_dict[f'{split_name}_SL']['labels'].append(split_data[i]['label_hard'])

            data_dict[f'{split_name}_SL']['soft_labels'].append(split_data[i]['label_soft'])
            soft_label_for_noisy = [0.0] * num_labels
            soft_label_for_noisy[split_data[i]['label_noisy']] = 1.0
            data_dict[f'{split_name}_NL']['soft_labels'].append(soft_label_for_noisy)
            soft_label_for_hard = [0.0] * num_labels
            if split_data[i]['label_hard'] >= 0:
                soft_label_for_hard[split_data[i]['label_hard']] = 1.0
            data_dict[f'{split_name}_HL']['soft_labels'].append(soft_label_for_hard)

    # Save data_dict as pickle file
    file_suffix = get_gen_file_suffix(args)
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset)
    check_path(preprocessed_path)
    if 'zerogen' in args.generation_type:
        for split in ['train', 'valid']:
            with open(os.path.join(preprocessed_path, f'{split}_ZG_NL_{file_suffix}.pkl'), 'wb') as f:
                pickle.dump(data_dict[f'{split}_NL'], f)
            print(f"saved {split}_ZG_NL_{file_suffix}.pkl")
            if args.gen_relabel != 'none':
                with open(os.path.join(preprocessed_path, f'{split}_ZG_SL_{file_suffix}.pkl'), 'wb') as f:
                    pickle.dump(data_dict[f'{split}_SL'], f)
                print(f"saved {split}_ZG_SL_{file_suffix}.pkl")
                with open(os.path.join(preprocessed_path, f'{split}_ZG_HL_{file_suffix}.pkl'), 'wb') as f:
                    pickle.dump(data_dict[f'{split}_HL'], f)
                print(f"saved {split}_ZG_HL_{file_suffix}.pkl")
    elif 'unigen' in args.generation_type:
        preprocessed_path = os.path.join(args.preprocess_path, args.task) # Unigen is not dataset-specific
        check_path(preprocessed_path)
        for split in ['train', 'valid']:
            with open(os.path.join(preprocessed_path, f'{split}_UG_NL_{file_suffix}.pkl'), 'wb') as f:
                pickle.dump(data_dict[f'{split}_NL'], f)
            print(f"saved {split}_UG_NL_{file_suffix}.pkl")
            if args.gen_relabel != 'none':
                with open(os.path.join(preprocessed_path, f'{split}_UG_SL_{file_suffix}.pkl'), 'wb') as f:
                    pickle.dump(data_dict[f'{split}_SL'], f)
                print(f"saved {split}_UG_SL_{file_suffix}.pkl")
                with open(os.path.join(preprocessed_path, f'{split}_UG_HL_{file_suffix}.pkl'), 'wb') as f:
                    pickle.dump(data_dict[f'{split}_HL'], f)
                print(f"saved {split}_UG_HL_{file_suffix}.pkl")


def process_output(output_text: str) -> str:
    """Process output from causal LM models (GPT-2 style) - split by ': "' pattern."""
    try:
        output_text = output_text.split(": \"")[1]
    except IndexError:
        return ""

    # if the sentence has \n, delete it and words after it
    if "\n" in output_text:
        output_text = output_text.split("\n")[0]
    if "\"" in output_text:
        output_text = output_text.split("\"")[0]

    return output_text.strip()

def process_output_instruction(output_text: str) -> str:
    """Process output from instruction-tuned models - the prompt is already stripped."""
    output_text = output_text.strip()

    # Remove surrounding quotes if present
    if output_text.startswith('"'):
        output_text = output_text[1:]
    if output_text.endswith('"'):
        output_text = output_text[:-1]

    # if the sentence has \n, delete it and words after it
    if "\n" in output_text:
        output_text = output_text.split("\n")[0]

    return output_text.strip()
