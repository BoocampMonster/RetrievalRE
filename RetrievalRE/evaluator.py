from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics import f1_score

from .dataset import KlueDataset
from .open_book_data_store import OpenBookDataStore
from .special_tokens import SPECIAL_ENTITY_MARKERS, get_relation_labels

import pandas as pd

class Evaluator:
    def __init__(self, args):
        self.args = args
        
        self.args = args
        self.device = "cuda:0" if args.use_gpu else "cpu"
        
        # Get relations in KLUE training dataset
        relations = KlueDataset.load_relations(
            data_path=args.data_path,
            data_fn=args.train_data_fn,
            cache_path=args.cache_path,
        )
        
        # Verbalization
        relation_labels = get_relation_labels(num_labels=len(relations))
        relation_label_map = {r: l for r, l in zip(relations, relation_labels)}
        self.label_relation_map = {l: r for l, r in zip(relation_labels, relations)}
             
        # Load tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        self.relation_ids = self.tokenizer.additional_special_tokens_ids[-30:]
        
        # Load model and resize token embeddings
        self.model = AutoModelForMaskedLM.from_pretrained(args.model_path)
        self.model = self.model.to(self.device)
        
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
                relation_label_map=relation_label_map,
                max_seq_length=args.max_seq_length,
                special_entity_markers=SPECIAL_ENTITY_MARKERS
            ),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.train_num_workers
        )
        
    def run(self):
        score = self.evaluate()
        
        print(f"Evaluation F1 Score Result : {score:.2f}")
            
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
        
    def calc_f1_score(self, true_y, pred_y):
        return f1_score(true_y, pred_y, average="micro", labels=self.relation_ids[-29:])
    