import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer, BertTokenizer
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import log_loss
import torch.nn.functional as F

df = pd.read_csv('data/train.csv')

warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

model_nm = 'microsoft/deberta-v3-base'
tokz = AutoTokenizer.from_pretrained(model_nm)
sep = tokz.sep_token
df['inputs'] = df.discourse_type + sep +df.discourse_text
new_label = {"discourse_effectiveness": {"Ineffective": 0, "Adequate": 1, "Effective": 2}}
df = df.replace(new_label)
df = df.rename(columns = {"discourse_effectiveness": "label"})
ds = Dataset.from_pandas(df)
def tok_func(x): return tokz(x["inputs"], truncation=True)
tok_func(ds[0])

inps = "discourse_text","discourse_type"
tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','discourse_id','essay_id'))

essay_ids = df.essay_id.unique()
np.random.seed(42)
np.random.shuffle(essay_ids)

val_prop = 0.2
val_sz = int(len(essay_ids)*val_prop)
val_essay_ids = essay_ids[:val_sz]

is_val = np.isin(df.essay_id, val_essay_ids)
idxs = np.arange(len(df))
val_idxs = idxs[ is_val]
trn_idxs = idxs[~is_val]

dds = DatasetDict({"train":tok_ds.select(trn_idxs),
             "test": tok_ds.select(val_idxs)})

def get_dds(df, train=True):
    ds = Dataset.from_pandas(df)
    to_remove = ['discourse_text','discourse_type','inputs','discourse_id','essay_id']
    tok_ds = ds.map(tok_func, batched=True, remove_columns=to_remove)
    if train:
        return DatasetDict({"train":tok_ds.select(trn_idxs), "test": tok_ds.select(val_idxs)})
    else: 
        return tok_ds

lr,bs = 8e-5,16
wd,epochs = 0.01,4

def score(preds): return {'log loss': log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}
def get_trainer(dds):
    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine',
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=wd, report_to='none')
    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=3)
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokz, compute_metrics=score)

trainer = get_trainer(dds)

trainer.train()

