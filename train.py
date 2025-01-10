import os
import math
import json
import torch
import argparse

from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from utils import get_bnb_config, get_prompt
from sklearn.model_selection import train_test_split
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="zake7749/gemma-2-2b-it-chinese-kyara-dpo",
        help="model",
    )
    
    parser.add_argument("--train", type=str,
                        default="./data/train.json",
                        help="training data")

    parser.add_argument("--train_val_split", type=float,
                        default=0.0,
                        help="train_val_split")

    parser.add_argument("--max_length", type=int,
                        default=512,
                        help="max_length")

    parser.add_argument("--num_train_epochs", type=int,
                        default=5,
                        help="num_train_epochs")

    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=1,
                        help="gradient_accumulation_steps")
    
    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=1,
                        help="per_device_train_batch_size")

    parser.add_argument("--learning_rate", type=float,
                        default=3e-5,
                        help="learning_rate")

    parser.add_argument("--lr_scheduler_type", type=str,
                        default="linear",
                        help="lr_scheduler_type")

    parser.add_argument("--num_warmup_steps", type=int,
                        default=300,
                        help="num_warmup_steps")
    
    parser.add_argument("--weight_decay", type=float,
                        default=1e-4,
                        help="weight_decay")

    parser.add_argument("--lora_rank", type=int,
                        default=None,
                        help="lora_rank")

    parser.add_argument("--lora_alpha", type=int,
                        default=16,
                        help="lora_alpha")

    parser.add_argument("--lora_dropout", type=float,
                        default=0.1,
                        help="dropout")

    parser.add_argument("--checkpointing_steps", type=int,
                        default=1000,
                        help="checkpointing_steps")

    parser.add_argument("--resume_from_checkpoint", type=bool,
                        default=False,
                        help="resume_from_checkpoint")

    parser.add_argument("--output_dir", type=str,
                        default=None,
                        help="output_dir")

    parser.add_argument("--seed", type=int,
                        default=42,
                        help="seed")

    return parser.parse_args()

def main():
    args = parse_args()
    
    dataset = {}
    with open(args.train, "r") as f:
        data = json.load(f)

    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        dataset["train"], dataset["validation"]  = train_test_split(data, test_size=args.train_val_split, random_state=args.seed)
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(args.seed)
    
    bnb_config = get_bnb_config()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    def pad_or_truncate(data, max_length, padding_token=0):
        if max_length >= len(data):
            return data + [padding_token] * (max_length - len(data))
        else:
            return data[:max_length]

    def preprocess_function(data, train, max_length):
        ids = [sample["id"] for sample in data]
        instructions = [get_prompt(sample["instruction"]) for sample in data]
        tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
        
        processed_data = []
        if train:
            outputs = [sample["output"] for sample in data]
            tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
            for i in range(len(data)):
                instructions_input_ids = [tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
                outputs_input_ids = tokenized_outputs["input_ids"][i] + [tokenizer.eos_token_id]
                processed_data_input_ids =  instructions_input_ids + outputs_input_ids
                processed_data.append(
                    {
                        "id": ids[i],
                        "input_ids": pad_or_truncate(processed_data_input_ids, max_length),
                        "attention_mask": pad_or_truncate([1] * len(processed_data_input_ids), max_length),
                        "labels": pad_or_truncate([-100] * len(instructions_input_ids) + outputs_input_ids, max_length),
                        "output_mask": pad_or_truncate([0] * len(instructions_input_ids) + [1] * len(outputs_input_ids), max_length),
                    }
                )
        else:
            processed_data_input_ids = [tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
            processed_data.append(
                {
                    "id": ids[i],
                    "input_ids": processed_data_input_ids,
                    "attention_mask": [1] * len(processed_data_input_ids),
                    "prompt": instructions[i],
                }
            )
        return processed_data
    

    with accelerator.main_process_first():
        train_dataset = preprocess_function(dataset["train"], train=True, max_length=args.max_length)
        
        if args.train_val_split > 0.0:
            valid_dataset = preprocess_function(dataset["validation"], train=True, max_length=args.max_length)
    
    def collate_func(data: list) -> dict:
        data_list_dict = {k: [dic[k] for dic in data] for k in data[0]}

        data_tensor_dict = {
            k: v if isinstance(v[0], str) else torch.tensor(v)
            for k, v in data_list_dict.items()
        }
        
        return data_tensor_dict
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_func, batch_size=args.per_device_train_batch_size
    )
    if args.train_val_split > 0.0:
        valid_dataloader = DataLoader(
            valid_dataset, shuffle=True, collate_fn=collate_func, batch_size=args.per_device_train_batch_size
        )
    
    
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    max_train_steps = 0
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps
        if overrode_max_train_steps
        else max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    checkpointing_steps = args.checkpointing_steps
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    progress_bar.update(completed_steps)
    
    accelerator.print("Start Training!")
    train_step_loss = []
    valid_step_loss = []
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                train_step_loss.append(loss.detach().float().item())
                # We keep track of the loss at each epoch
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= max_train_steps:
                break
            
        model.eval()
        for step, batch in enumerate(valid_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                valid_step_loss.append(loss.detach().float().item())
    
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )

        with open(os.path.join(args.output_dir, "train_step_loss.json"), "w") as f:
                json.dump(train_step_loss, f)
        with open(os.path.join(args.output_dir, "valid_step_loss.json"), "w") as f:
                json.dump(valid_step_loss, f)
        
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
    
if __name__ == "__main__":
    main()
