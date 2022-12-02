import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import torch
import numpy as np
import argparse
from .trainer import Trainer
from .builder import Builder
from .evaluator import Evaluator

from .file_io import *

import wandb
from omegaconf import OmegaConf

def fix_seed(random_seed=42):
    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)

args = None
    
def init_dirs():
    make_dir(args.cache_path)
    make_dir(args.model_path)

def train():
    trainer = Trainer(args=args)
    trainer.run()
    
def wandb_sweep():
    with wandb.init() as run:
        # update any values not set by sweep
        # run.config.setdefaults(config)
        for k, v in run.config.items():
            OmegaConf.update(args, k, v)
        train()
    
if __name__ == "__main__":
    fix_seed()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--run_mode", default="train")
    
    parser.add_argument("--plm", default="klue/roberta-large")
    # parser.add_argument("--pretrained_model", default="/home/chnaaam/chnaaam/KnowPrompt/pretrained_model")
    
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--train_data_fn", default="klue-re-v1.1_train.json")
    parser.add_argument("--valid_data_fn", default="klue-re-v1.1_dev.json")
    parser.add_argument("--test_data_fn", default="klue-re-v1.1_test.json")
    parser.add_argument("--build_data_fn", default="klue-re-v1.1_build.json")
    
    parser.add_argument("--cache_path", default="./cache")
    parser.add_argument("--cache_model_path", default="./cache")
    parser.add_argument("--model_path", default="./model")
    
    parser.add_argument("--use_gpu", default=True)
    parser.add_argument("--use_fp16", default=True)
    parser.add_argument("--max_seq_length", default=256)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-5)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--train_num_workers", default=0)
    parser.add_argument("--valid_num_workers", default=0)
    
    parser.add_argument("--topk", default=12)
    parser.add_argument("--logit_ratio", default=0.4)
    
    parser.add_argument('--wandb', type=str, default='init')
    
    args = parser.parse_args()
    
    init_dirs()
    
    if args.run_mode == "train":
        # wandb 설정을 해주지 않으면 오류가 납니다
        wandb_config = OmegaConf.load(f'./train/wandb_{args.wandb}.yaml')
        
        print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')
        
        wandb.login()
        
        wandb.config = args
        
        if wandb_config.get('sweep'):
            sweep_config = OmegaConf.to_object(wandb_config.sweep)
            sweep_id = wandb.sweep(
                    sweep=sweep_config,
                    entity=wandb_config.entity,
                    project=wandb_config.project)
            wandb.agent(sweep_id=sweep_id, function=wandb_sweep, count=wandb_config.count)
            
        else:
            wandb.init(
                    entity=wandb_config.entity,
                    project=wandb_config.project,
                    group=wandb_config.group,
                    name=wandb_config.experiment)
            train()
        
    elif args.run_mode == "build":
        builder = Builder(args=args)
        builder.run()
        
    elif args.run_mode == "evaluate":
        evaluator = Evaluator(args=args)
        evaluator.run()
        