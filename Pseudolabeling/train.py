
########################################
# Directory settings
########################################
import os

INPUT_DIR = './'
OUTPUT_DIR = './out/all/55/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

########################################
# CFG
########################################
class CFG:
    apex=True
    print_freq=2500
    num_workers=0
    model="microsoft/deberta-v3-small"
    scheduler='cosine'
    batch_scheduler=True
    num_cycles=0.45
    num_warmup_steps=0
    epochs=2
    encoder_lr=2e-5
    decoder_lr=2e-5
    eps=2e-6
    betas=(0.9, 0.999)
    batch_size=96
    max_len=152
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    n_fold=10
    trn_fold=[0]
    model_path=None

########################################
# Library
########################################
import gc
import time
import math
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_score1(y_true, y_pred):
    score = accuracy_score(y_true, y_pred)
    return score

def get_score2(y_true, y_pred):
    score = r2_score(y_true, y_pred)
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

LOGGER = get_logger()

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

def mappingfn1(x):
    if x < 0.01:
        return 0
    else:
        return 1

def mappingfn2(x):
    if x < 0.54:
        return 0
    if x > 0.59:
        return 1

def zeroes(x):
    if x == 0:
        return 0
        
seed_everything(seed=CFG.seed)

def excluder(x):
    if x == 1:
        return x


ts1 = [
       'severe_toxicity', 'obscene',
       'identity_attack', 'insult', 'threat', 'asian', 'atheist', 'bisexual',
       'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
       'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',
       'jewish', 'latino', 'male', 'muslim', 'other_disability',
       'other_gender', 'other_race_or_ethnicity', 'other_religion',
       'other_sexual_orientation', 'physical_disability',
       'psychiatric_or_mental_illness', 'transgender', 'white'
       ]

ts = {
       'race': ['black', 'white', 'asian', 'other_race_or_ethnicity', 'jewish', 'latino'],
'gender': ['male', 'female', 'other_gender', 'transgender'],
'religion': ['atheist', 'hindu', 'christian', 'other_religion', 'buddhist', 'muslim'],
'disability': ['intellectual_or_learning_disability','physical_disability','psychiatric_or_mental_illness'],
'sexuality': ['heterosexual', 'other_sexual_orientation', 'homosexual_gay_or_lesbian', 'bisexual', 'transgender']
}


########################################
# Dataset
########################################
def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length",
                           return_offsets_mapping=False,
                           truncation=True)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['comment_text'].values
        self.labels = df['effectiveness'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.long)
        return inputs, label



########################################
# Model
########################################
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

# Helper functions

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
        #print(y_preds.view(-1, 2), labels.view(-1))
        #loss = criterion(y_preds.view(-1, 2), labels.view(-1))
        loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()

    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        preds.append(torch.max(y_preds, 1)[1].to('cpu'))
        batch_size = labels.size(0)
        losses.update(loss.item(), batch_size)
        
    predictions = []
    for pred in preds:
        prediction = pred.tolist()
        predictions.extend(prediction)

    return predictions

class EarlyStopping():
    def __init__(self, tolerance=3, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss, best_score, score):
        if (validation_loss - train_loss) > self.min_delta and best_score > score:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.counter = 0
                self.early_stop = True

def lossweights(ylabels):
    class_weights = compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(ylabels),
                                            y = ylabels                                                    
                                        )
    #class_weights = dict(zip(np.unique(train_labels), class_weights))
    print(class_weights)
    return torch.tensor(class_weights, dtype=torch.long).cuda()



def train_loop(folds, fold):
    if fold == 2:
            CFG.encoder_lr=1e-5
            CFG.decoder_lr=1e-5
            
    LOGGER.info(f"========== fold: {fold} training ==========")

    # loader

    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_folds.to_csv('validdf.csv', index=False)
    valid_labels = valid_folds['effectiveness'].values
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=round(CFG.batch_size * 1.2),
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # model & optimizer

    model = CustomModel(CFG, config_path=None, pretrained=False)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=CFG.weight_decay):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    

    # scheduler

    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int((len(train_folds) / CFG.batch_size * CFG.epochs) / CFG.gradient_accumulation_steps)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)


    # loop

    sample_weight = torch.Tensor([1, 1]).to(torch.float16).cuda()
    sample_weight_val = torch.Tensor([1, 1]).cuda()
    criterion = nn.CrossEntropyLoss(reduction="mean", weight=sample_weight)
    criterion_val = nn.CrossEntropyLoss(reduction="mean", weight=sample_weight_val)
    #criterion = nn.CrossEntropyLoss(reduction="mean")
    #criterion_val = nn.CrossEntropyLoss(reduction="mean")
    best_score = 0.
    

    for epoch in range(CFG.epochs):
                
        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        #eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion_val, device)
        
        #scoring
        score = get_score1(valid_labels, predictions)
        f1 = f1_score(valid_labels, predictions, average='binary')
        print(classification_report(valid_labels, predictions))

        elapsed = time.time() - start_time
        
        avg_loss = avg_loss * CFG.gradient_accumulation_steps
        avg_val_loss = avg_val_loss * CFG.gradient_accumulation_steps
        

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Accuracy: {score:.4f} - F1: {f1:.4f}')
                
        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} - F1: {f1:.4f}')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
        else:
            state = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                       map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            
    predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    valid_folds['preds'] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds



if __name__ == '__main__':

    train = pd.read_csv(INPUT_DIR+'train.csv')
    train.fillna(0, inplace=True)
    train['effectiveness'] = train['target'].apply(mappingfn2)
    train['effectiveness'] = train['effectiveness'].apply(excluder)
    train.dropna(inplace=True)


    train["fold"] = -1
    train = train.sample(frac=1).reset_index(drop=True)
    y = train.effectiveness.values
    kf = model_selection.StratifiedKFold(n_splits=CFG.n_fold)

    for f, (t_, v_) in enumerate(kf.split(X=train, y=y)):
        train.loc[v_, 'fold'] = f
        train.head()

    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    CFG.tokenizer = tokenizer
    
    
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(train, fold)
            oof_df = pd.concat([oof_df, _oof_df])
    oof_df = oof_df.reset_index(drop=True)



