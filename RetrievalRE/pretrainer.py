import os
from tqdm import tqdm
from distutils.dir_util import copy_tree

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoModelForMaskedLM, AutoTokenizer
from accelerate import Accelerator
from sklearn.metrics import f1_score

from .dataset import KlueDataset
from .special_tokens import SPECIAL_ENTITY_MARKERS, get_relation_labels

from .open_book_data_store import OpenBookDataStore

import time
import wandb

from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np

import pandas as pd

# https://github.com/wonjun-dev/AI-Paper-Reproduce/blob/master/simCSE-Pytorch/pretrain.py
class SimCSELoss(nn.Module):
    def __init__(self, batch_size, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()    # negative pair를 indexing 하는 마스크입니다. (자기 자신(대각 성분)을 제외한 나머지) 

    def calc_sim_batch(self, a, b):
        reprs = torch.cat([a, b], dim=0)
        return F.cosine_similarity(reprs.unsqueeze(1), reprs.unsqueeze(0), dim=2)   # 두 representation의 cosine 유사도를 계산합니다.
    
    def calc_align(self, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean().detach()  # 두 representation의 alignment를 계산하고 반환합니다.

    def calc_unif(self, x, t=2):
        sp_pdist = torch.pdist(x, p=2).pow(2)
        return sp_pdist.mul(-t).exp().mean().log().detach()  # 미니 배치 내의 represenation의 uniformity를 계산하고 반환합니다.
    
    def forward(self, proj_1, proj_2):
        batch_size = proj_1.shape[0]
        if batch_size != self.batch_size:
            mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float() # 에폭 안에서 마지막 미니 배치를 위해서 마스크를 새롭게 정의 합니다.
        else:
            mask = self.mask
            
        z_i = F.normalize(proj_1, p=2, dim=1)   # 모델의 [CLS] represenation을 l2 nomalize 합니다.
        z_j = F.normalize(proj_2, p=2, dim=1)

        sim_matrix = self.calc_sim_batch(z_i, z_j)  # 배치 단위로 두 representation의 cosine 유사도를 계산합니다.

        sim_ij = torch.diag(sim_matrix, batch_size) # sim_matrix에서 positive pair의 위치를 인덱싱 합니다. (대각 성분에서 배치 사이즈만큼 떨어져 있습니다.)
        sim_ji = torch.diag(sim_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = mask.to(sim_matrix.device) * torch.exp(sim_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))  # constrastive loss
        loss = torch.sum(all_losses) / (2 * batch_size) # 샘플 갯수로 나누어 평균 내줍니다.

        lalign = self.calc_align(z_i, z_j)
        lunif = (self.calc_unif(z_i[:batch_size//2]) + self.calc_unif(z_i[batch_size//2:])) / 2

        return loss, lalign, lunif

'simCSE loss(unsup.)' : epoch_loss / steps,
                'measure_align' : lalign,
                'measure_unif' : lunif,

class Trainer:
    def __init__(self, args):
        self.args = args
        
        wandb.login()
        
        wandb.init(project='retrieval-re', group='label-30', name='trial-3', entity='nlp6')
        
        wandb.config.update(args)
        
        # Get relations in KLUE training dataset
        relations = KlueDataset.load_relations(
            data_path=args.data_path,
            data_fn=args.train_data_fn,
            cache_path=args.cache_path,
        )
        
        # Verbalization
        relation_labels = get_relation_labels(num_labels=len(relations))
        self.relation_label_map = {r: l for r, l in zip(relations, relation_labels)}
        self.label_relation_map = {l: r for l, r in zip(relation_labels, relations)}
             
        # Load tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm)
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": SPECIAL_ENTITY_MARKERS + relation_labels
        })
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.args.model_path)
        
        self.relation_ids = self.tokenizer.additional_special_tokens_ids[-30:]
        
        # Load model and resize token embeddings
        self.model = AutoModelForMaskedLM.from_pretrained(args.plm)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        train_dataset = KlueDataset(
            tokenizer=self.tokenizer,
            data_path=args.data_path,
            data_fn=args.train_data_fn,
            cache_path=args.cache_path,
            cache_fn=f"{self.args.plm.replace('/', '_')}.cache.train",
            relation_label_map=self.relation_label_map,
            max_seq_length=args.max_seq_length,
            special_entity_markers=SPECIAL_ENTITY_MARKERS
        )
        
        train_label = train_dataset.get_labels()
        # https://github.com/wonjun-dev/AI-Paper-Reproduce/blob/master/simCSE-Pytorch/pretrain.py
        # long tail distribution 를 고려해서 label이 균등한 미니배치를 구성하기 위해 weighted sampler 정의
        class_sample_count = np.array([len(np.where(train_label == t)[0]) for t in np.unique(train_label)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t - self.relation_ids[-30]] for t in train_label])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        # Load data loader
        self.train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            # shuffle=True,
            sampler=sampler,
            pin_memory=True,
            num_workers=args.train_num_workers
        )
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=float(args.lr))
        self.criterion = nn.CrossEntropyLoss()
        
        self.accelerator = Accelerator(
            cpu=False if self.args.use_gpu else True,
            fp16=args.use_fp16
        )
        
        self.model, self.optimizer, self.train_data_loader = self.accelerator.prepare(
            self.model, 
            self.optimizer, 
            self.train_data_loader
        )
        
        self.device = self.accelerator.device
        
    def run(self):
        max_score, max_score_epoch = 0, 0
        avg_train_loss = 0.0
        
        for epoch in range(self.args.epochs):
            standard_time = time.time()
            avg_train_loss = self.train(epoch)
            
            wandb.log({'epoch' : epoch, 'runtime(Min)' : (time.time() - standard_time) / 60})
            
            if max_score < valid_score:
                max_score = valid_score
                max_score_epoch = epoch
            
            # Save trained model
            self.model.save_pretrained(
                os.path.join(
                    self.args.cache_model_path,
                    f"e-{epoch}.plm-{self.args.plm.replace('/', '_')}.train-loss-{avg_train_loss:.4f}"
                )
            )
        
        max_score_model_dir = ""
        for model_dir in os.listdir(self.args.cache_model_path):
            if model_dir.startswith(f"e-{max_score_epoch}"):
                max_score_model_dir = model_dir
                
        copy_tree(
            os.path.join(self.args.cache_model_path, max_score_model_dir), 
            os.path.join(self.args.model_path)
        )
        
        
    def train(self, epoch):
        self.model.train()
        
        losses = []
        avg_loss = 0.0
        
        progress_bar = tqdm(self.train_data_loader)
        for batch in progress_bar:
            progress_bar.set_description(f"[Training] Epoch : {epoch}, Avg Loss : {avg_loss:.4f}")
            
            ids, inputs, labels = batch
            
            logits = self.model(**inputs).logits
            
            mask_logits = self.get_mask_logits(
                logits=logits,
                mask_idxes=(inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            )
            
            loss = self.criterion(mask_logits, labels)
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            
            losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            
            wandb.log({'train_loss': avg_loss})
            
        return sum(losses) / len(losses)
    