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
        
        self.valid_data_loader = DataLoader(
            dataset=KlueDataset(
                tokenizer=self.tokenizer,
                data_path=args.data_path,
                data_fn=args.valid_data_fn,
                cache_path=args.cache_path,
                cache_fn=f"{self.args.plm.replace('/', '_')}.cache.valid",
                relation_label_map=self.relation_label_map,
                max_seq_length=args.max_seq_length,
                special_entity_markers=SPECIAL_ENTITY_MARKERS
            ),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.train_num_workers
        )
        
        # Load open book data store
        self.data_store = OpenBookDataStore(self.model.config.hidden_size)
        self.data_store.load(model_path=self.args.model_path)
        
        # Load data loader
        self.eval_data_loader = DataLoader(
            dataset=KlueDataset(
                tokenizer=self.tokenizer,
                data_path=args.data_path,
                data_fn=args.test_data_fn,
                cache_path=args.cache_path,
                cache_fn=f"{self.args.plm.replace('/', '_')}.cache.test",
                relation_label_map=self.relation_label_map,
                max_seq_length=args.max_seq_length,
                special_entity_markers=SPECIAL_ENTITY_MARKERS
            ),
            batch_size=args.batch_size,
            shuffle=True,
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
        avg_train_loss, avg_valid_loss, valid_score = 0.0, 0.0, 0.0
        
        for epoch in range(self.args.epochs):
            standard_time = time.time()
            avg_train_loss = self.train(epoch)
            avg_valid_loss, valid_score = self.valid(epoch)
            test_score = self.evaluate()
            
            print(f"Validation F1 Score Result : {valid_score:.2f}")
            print(f"Evaluation F1 Score Result : {test_score:.2f}")
            
            wandb.log({'epoch' : epoch, 'runtime(Min)' : (time.time() - standard_time) / 60})
            wandb.log({'epoch' : epoch, 'val_loss':avg_valid_loss})
            wandb.log({'epoch' : epoch, 'val_score': valid_score})
            
            if max_score < valid_score:
                max_score = valid_score
                max_score_epoch = epoch
            
            # Save trained model
            self.model.save_pretrained(
                os.path.join(
                    self.args.cache_model_path,
                    f"e-{epoch}.plm-{self.args.plm.replace('/', '_')}.train-loss-{avg_train_loss:.4f}.valid-loss-{avg_valid_loss:.4f}.score-{valid_score:.2f}.score-{test_score:.2f}"
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
    
    def valid(self, epoch):
        self.model.eval()
        
        losses = []
        avg_loss = 0.0
        
        output_pred = []
        output_label = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.valid_data_loader)
            for batch in progress_bar:
                progress_bar.set_description(f"[Validation] Epoch : {epoch}, Avg Loss : {avg_loss:.4f}")
                ids, inputs, labels = batch
            
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                logits = self.model(**inputs).logits
                
                mask_logits = self.get_mask_logits(
                    logits=logits,
                    mask_idxes=(inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                )
                
                loss = self.criterion(mask_logits, labels)
                losses.append(loss.item())
                
                # f1_score = self.calc_f1_score(
                #     true_y=labels.tolist(), 
                #     pred_y=torch.argmax(torch.softmax(mask_logits, dim=-1), dim=-1).tolist()
                # )
                
                avg_loss = sum(losses) / len(losses)
                
                pred_y=torch.argmax(torch.softmax(mask_logits.detach(), dim=-1), dim=-1).cpu().tolist()
            
                output_pred.extend(pred_y)
                output_label.extend(labels.detach().cpu().tolist())
        
        score = self.calc_f1_score(
            true_y=output_label, 
            pred_y=output_pred
        ) * 100
        
        return avg_loss, score
    
    def test(self, epoch):
        self.model.eval()
        
        losses = []
        avg_loss = 0.0
        
        output_pred = []
        output_label = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_data_loader)
            for batch in progress_bar:
                progress_bar.set_description(f"[Evaluation] Epoch : {epoch}, Avg Loss : {avg_loss:.4f}")
                ids, inputs, labels = batch
            
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                logits = self.model(**inputs).logits
                
                mask_logits = self.get_mask_logits(
                    logits=logits,
                    mask_idxes=(inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                )
                
                loss = self.criterion(mask_logits, labels)
                losses.append(loss.item())
                
                # f1_score = self.calc_f1_score(
                #     true_y=labels.tolist(), 
                #     pred_y=torch.argmax(torch.softmax(mask_logits, dim=-1), dim=-1).tolist()
                # )
                
                avg_loss = sum(losses) / len(losses)
                
                pred_y=torch.argmax(torch.softmax(mask_logits.detach(), dim=-1), dim=-1).cpu().tolist()
            
                output_pred.extend(pred_y)
                output_label.extend(labels.detach().cpu().tolist())
        
        score = self.calc_f1_score(
            true_y=output_label, 
            pred_y=output_pred
        ) * 100
        
        return avg_loss, score
    
    def evaluate(self):
        self.model.eval()
        
        score = 0.0
        
        output_id = []
        output_pred = []
        output_prob = []
        output_label = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.eval_data_loader)
            for batch in progress_bar:
                ids, inputs, labels = batch
            
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                output = self.model(**inputs, output_hidden_states=True)
                logits = output.logits
                hidden_state = output.hidden_states[-1]
                
                mask_logits = self.get_mask_logits(
                    logits=logits,
                    mask_idxes=(inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                )
                
                knn_logits = self.get_knn_logits(
                    hidden_state=self.get_mask_hidden_state(
                        hidden_state=hidden_state,
                        mask_idxes=(inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                    ),
                    topk=self.args.topk
                )
                
                logits = self.args.logit_ratio * knn_logits + (1 - self.args.logit_ratio) * mask_logits
                # logits = mask_logits
                
                # prob = torch.nn.functional.pad(torch.softmax(logits[:, -29:], dim=-1), (1, 0), "constant", 0)
                prob = torch.softmax(logits[:, -30:], dim=-1)
                
                pred_y = torch.argmax(prob, dim=-1) + (logits.shape[-1] - 30)
                
                output_id.extend(ids.detach().cpu().tolist())
                output_pred.extend(pred_y.detach().cpu().tolist())
                output_prob.extend(prob.detach().cpu().tolist())
                output_label.extend(labels.detach().cpu().tolist())
        
        score = self.calc_f1_score(
            true_y=output_label, 
            pred_y=output_pred
        ) * 100
        
        pred_answer = [self.label_relation_map[self.tokenizer.convert_ids_to_tokens(label)] for label in output_pred]
        
        output = pd.DataFrame({'id':output_id,'pred_label':pred_answer,'probs':output_prob,})
        output.sort_values('id', inplace=True)
        output.to_csv('./submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
        
        return score
    
    def get_mask_logits(self, logits, mask_idxes):
        return logits[torch.arange(logits.shape[0]), mask_idxes]
    
    def calc_f1_score(self, true_y, pred_y):
        return f1_score(true_y, pred_y, average="micro", labels=self.relation_ids[-29:])
            
    def get_mask_hidden_state(self, hidden_state, mask_idxes):
        return hidden_state[torch.arange(hidden_state.shape[0]), mask_idxes]
    
    def get_knn_logits(self, hidden_state, topk):
        knn_logits = torch.full((hidden_state.shape[0], len(self.tokenizer)), 1000.0).to(self.device)
        
        distances, indexes = self.data_store.search(
            hidden_state=hidden_state.cpu().numpy(),
            topk=topk
        )
        distances = torch.from_numpy(distances).to(self.device)
        
        for i in range(hidden_state.shape[0]):
            for j in range(topk):
                if knn_logits[i][self.data_store.get_label_from_index(indexes[i][j])] > distances[i][j]:
                    knn_logits[i][self.data_store.get_label_from_index(indexes[i][j])] = distances[i][j]
        
        if torch.sum(knn_logits) != 0.0:
            knn_logits = torch.softmax((-1) * knn_logits, dim=-1)
            
        return knn_logits
    