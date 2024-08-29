import torch
import wandb
import argparse
import emoji
import pandas as pd
from trainer import Trainer
from tqdm import trange
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import get_scheduler
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          ElectraForSequenceClassification, ElectraTokenizer,
                          ConvBertForSequenceClassification, ConvBertTokenizer,
                          RobertaForSequenceClassification,RobertaTokenizer)




# CONSTANTS
domains = ["politics", 'sports', 'healthcare', 'jobEducation', 'entertainment']
platform = "youtube"
device = 'cuda:0'
best_models = {
    'hateSpeech': 'hate-speech-robertatwitter-1',
    'genderBias': 'gender-bias-convbert-2',
    'linguisticBias': 'linguistic-bias-convbert-0',
    'racialBias': 'racial-bias-electra-2',
    'politicalBias': 'political-bias-convbert-4',
    'textLevelBias': 'text-level-bias-convbert-1'
}

def load_model(name, model_length=128):
    if "convbert" in name:
        convbert_tokenizer = ConvBertTokenizer.from_pretrained(
            'YituTech/conv-bert-base', model_max_length=model_length)
        convbert_model = ConvBertForSequenceClassification.from_pretrained(
            './saved_models/'+name, num_labels=2)
        learning_rate = 5e-5
        return convbert_model, convbert_tokenizer, learning_rate

    elif "robertatwitter" in name:
        roberta_twitter_tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base", model_max_length=model_length)
        roberta_twitter_model = AutoModelForSequenceClassification.from_pretrained(
            './saved_models/'+name, num_labels=2)
        learning_rate = 5e-5
        return roberta_twitter_model, roberta_twitter_tokenizer, learning_rate

    elif "electra" in name:
        electra_tokenizer = ElectraTokenizer.from_pretrained(
            'google/electra-base-discriminator', model_max_length=model_length)
        electra_model = ElectraForSequenceClassification.from_pretrained(
            'google/electra-base-discriminator', num_labels=2)
        learning_rate = 5e-5
        return electra_model, electra_tokenizer, learning_rate

    elif "roberta" in name:
        roberta_tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", model_max_length=model_length,use_fast=False)
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base', num_labels=2)
        learning_rate = 5e-5
        return roberta_model, roberta_tokenizer, learning_rate

    else:
        print(f'Model not found: {name}')
        raise ValueError

def tokenize_data(df, tokenizer):
    tokenized = []
    ids = []
    print("Tokenizing...")
    for i in tqdm(range(len(df))):
        try:
            # Removing emojis AND weird punctuations
            text = emoji.replace_emoji(df.iloc[i]["textDisplay"], '')
            text = text.replace('\"', '')
            # text = emoji.demojize(df.iloc[i]["textDisplay"])
            tok = tokenizer(text, padding="max_length", truncation=True)
            tok["input_ids"] = torch.tensor(tok["input_ids"])
            tok["attention_mask"] = torch.tensor(tok["attention_mask"])
            # tok["labels"] = torch.tensor(df.iloc[i]["label"])
            if "token_type_ids" in tok.keys():
                tok["token_type_ids"] = torch.tensor(tok["token_type_ids"])
            tokenized.append(tok)
        except:
            ids.append(i)
            continue
    return tokenized, ids

def evaluate(model, dl, device):
    num_steps = len(dl)
    progress_bar = tqdm(range(num_steps))
    predictions = []
    model.to(device)
    model.eval()
    
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1))
        progress_bar.update(1)

    predictions = torch.stack(predictions).cpu()
    return predictions

if __name__ == "__main__":
    print("Start evaluating...")
    for domain in domains:
        domain_data = pd.read_csv(f"./annotations/{platform}_{domain}.csv")
        domain_data = domain_data.drop(['Unnamed: 0'], axis=1)
        # HARD CODE[for healthcare]: 
        # dropped_ids = [1208, 1209, 1929, 1930, 1931, 2569] 
        # domain_data = domain_data.drop(dropped_ids)

        output_eval = True
        if output_eval:
            for k, item in best_models.items():
                # df = domain_data.copy(deep=True)
                model, tokenizer, _ = load_model(item)
                tokenized_df, dropped_ids = tokenize_data(domain_data, tokenizer)
                if len(dropped_ids) != 0:
                    domain_data = domain_data.drop(dropped_ids)

                # df = df.drop(dropped_ids)
                eval_dl = DataLoader(tokenized_df, batch_size=16)
                pred = evaluate(model, eval_dl, device)
                domain_data[f'{k}_pred'] = pred
            
            domain_data.to_csv(f"predicted-{platform}-{domain}.csv")
                
        
        
    