import os
import logging
import argparse
import torch
import math
import transformers
import copy
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from peft import PeftModel
from trainer import Trainer
from dataset import DataCollatorForSupervisedDataset,SupervisedDataset

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Finetune a LLM model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="NBatch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="NBatch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")


    parser.add_argument("--max_length", type=int, default=512,
                        help=(
                            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."))

    parser.add_argument("--model_name_or_path", type=str, 
                        #default="/server1/suguo/suguo/simple_thu_chatglm6b/thuglm/", 
                        default="/server1/suguo/suguo/DolphinGen/pretraining/chatglm-6b",
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--dataset_path", type=str, default='/server1/suguo/suguo/DolphinGen/data/zh_seed_tasks.json',
                        help="This name is the path to the dataset")
    parser.add_argument("--output_dir", type=str, default="/server1/suguo/suguo/DolphinGen/result_chatglm-6b", help="Where to store the final model.")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--steps_to_save", type=int, default=None, help="Num steps to save the checkpoint")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--steps_to_log", type=int, default=1, help="Num steps to log training info")
    parser.add_argument("--steps_to_evaluate", type=int, default=2, help="Num steps to log training info")

    parser.add_argument("--use_lora", action="store_true", default=True, help="Whether to use Lora for parameter efficient tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="Lora rank, only used if use_lora is True")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha, only used if use_lora is True")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Lora dropout, only used if use_lora is True")

    parser.add_argument("--from_ckpt", action="store_true", help="restore the model training process from a checkpoint")
    parser.add_argument("--ckpt_model_path", type=str, default=None, help="The path of the model to continue training.")

    parser.add_argument("--cuda_devices", type=str, default='0', help="visible cuda devices.")

    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--max_ckpts_to_keep", type=int, default=3, help="Number of checkpoints to keep")
    parser.add_argument("--save_final", action="store_true", help="save the final checkpoint")
    args = parser.parse_args()

    if args.max_train_steps is None and args.num_train_epochs is None:
        raise ValueError("At least one of parameters max_train_steps and num_train_epochs is not None.")

    return args

def get_logger(args, accelerator):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logfile = os.path.join(args.output_dir, "log")
        if accelerator.is_main_process:
            if os.path.exists(logfile):
                os.remove(logfile)
            # os.mknod(logfile)# unix
            fh = logging.FileHandler(logfile, mode='w')
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.ERROR)
    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    return logger

def get_dataset(args, tokenizer):
    '''
    :param args:
    :param accelerator:
    :return:
    '''  

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.dataset_path, max_length=args.max_length)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, pin_memory=True
    )

    eval_dataloader = None
    return train_dataloader, eval_dataloader

def get_model(args, accelerator):
    # Load pretrained model and tokenizer
    #
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if args.model_name_or_path:
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, gradient_checkpointing=True)
        if args.from_ckpt and args.ckpt_model_path is not None:
            logger.info("loading checkpoint model. checkpoint path:{}".format(args.ckpt_model_path))
            accelerator.print("loading checkpoint model. checkpoint path:{}".format(args.ckpt_model_path))
            model = PeftModel.from_pretrained(model, args.ckpt_model_path)
    else:
        logger.error("model_name_or_path cannot be None.")
        raise ValueError("model_name_or_path cannot be None.")
    if args.use_lora:
        logger.info("Using the Lora Training Model")
        accelerator.print("Using the Lora Training Model")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            # ['dense','dense_h_to_4h','dense_4h_to_h'] # 'query_key_value',
            target_modules=['query_key_value'],
        )
        model = get_peft_model(model, peft_config)
        logger.info(model.print_trainable_parameters())
        #accelerator.print(model.print_trainable_parameters())

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    return tokenizer, model


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    os.environ['NVIDIA_VISIBLE_DEVICES'] = args.cuda_devices
    accelerator = Accelerator()
    args.device = accelerator.device
    logger = get_logger(args, accelerator)
    tokenizer, model = get_model(args, accelerator)
    train_dataloader, eval_dataloader = get_dataset(args, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        args.steps_to_save = num_update_steps_per_epoch # 每个epoch保存一次
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )
    # accelerator.print(model)

    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        logger=logger,
        accelerator=accelerator,
        from_checkpoint=args.ckpt_model_path,
        tokenizer=tokenizer,
    )
    trainer.train()

if __name__ == "__main__":
    args = get_args()
    main(args)