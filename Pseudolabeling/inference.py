

import os

INPUT_DIR = ''
OUTPUT_DIR = ''
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

import torch
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CFG:
    num_workers=4
    path= INPUT_DIR
    config_path= INPUT_DIR + 'config.pth'
    model_path = INPUT_DIR + 'microsoft-deberta-v3.pth'
    model="microsoft/deberta-v3-small"
    batch_size=1000
    max_len=152
    seed=42


def get_score1(y_true, y_pred):
    score = accuracy_score(y_true, y_pred)
    return score


def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed=28):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text.lower()

    data = clean_special_chars(data, punct)
    return data
        


# Dataset

def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length",
                           return_offsets_mapping=False,
                           truncation = True)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs


# Model

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True, max_position_embeddings = 512)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model_path, config=self.config)
        else:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config) 
            
        self.dropout2 = nn.Dropout(0.3)
        self.hidden_lstm_size = cfg.max_len
        self.attention = nn.Sequential(
            nn.Linear(768, 386),
            nn.Tanh(),
            nn.Linear(386, 1),
            nn.Softmax(dim=1)
        )
        self.fc1 = nn.Linear(768, 2)
        
    def squash(self, inputs):
        weights = self.attention(inputs)
        sentence_embedding = torch.sum(inputs * weights, dim=1)
        return sentence_embedding
    
    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        sentence_embedding = self.squash(last_hidden_states)
        sentence_with_dropout = self.dropout2(sentence_embedding)
        linear_output = self.fc1(sentence_with_dropout)
        return linear_output

def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    for step, inputs in enumerate(test_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
            preds.append(torch.max(y_preds, 1)[1].to('cpu'))
    predictions = []
    for pred in preds:
        prediction = pred.tolist()
        predictions.extend(prediction)
    return predictions


if __name__ == '__main__':

    LOGGER = get_logger()
    seed_everything(seed=CFG.seed)

    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model)

    test = pd.read_csv('Reddit_negeative_comments.csv')
    test['text'] = test['text'].apply(preprocess)

    test_dataset = TestDataset(CFG, test)
    test_loader = DataLoader(test_dataset,
                            batch_size=CFG.batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)

    state = torch.load(CFG.model_path, map_location=torch.device('cpu'))

    model.load_state_dict(state['model'])

    prediction = inference_fn(test_loader, model, device)
    test['score'] = prediction
    test.to_csv('reddit_pseudo_labels_55.csv', index=False)


