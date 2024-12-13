import torch
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
import argparse
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BartForSequenceClassification, BartTokenizer,
                          ConvBertForSequenceClassification, ConvBertTokenizer,
                          ElectraForSequenceClassification, ElectraTokenizer,
                          GPT2ForSequenceClassification, GPT2Tokenizer,
                          RobertaForSequenceClassification,RobertaTokenizer)


"""
train.py
Training script for shallow models using the datasets library and a specified pre-trained model.
The script includes data preprocessing, model training, and evaluation.
Functions:
    preprocess_function(examples): Tokenizes input text data.
    collate_fn(batch): Collates a batch of data for DataLoader.
Main Function:
    Parses command-line arguments for training configuration.
    Loads the specified dataset and splits it into training and evaluation sets.
    Loads the model, tokenizer, and optimizer.
    Trains the model for a specified number of epochs.
    Evaluates the model and prints a classification report.
Arguments:
    --device: Device to use for training (default: 'cuda:4'). Modify this to your own device if needed.
    --task_name: Name of the dataset to use (required).
    --model_name: Name of the pre-trained model (default: 'convbert').
    --batch_size: Batch size for training and evaluation (default: 32).
    --learning_rate: Learning rate for the optimizer (default: 5e-5).
    --max_length: Maximum length of the input sequence (default: 128).
    --num_epochs: Number of epochs to train the model (default: 3).
"""


def modelspecifications(name, model_length=128):
    if name == "convbert":
        convbert_tokenizer = ConvBertTokenizer.from_pretrained(
            'YituTech/conv-bert-base', model_max_length=model_length, return_tensors="pt")
        convbert_model = ConvBertForSequenceClassification.from_pretrained(
            'YituTech/conv-bert-base', num_labels=2)
        learning_rate = 5e-5
        return convbert_model, convbert_tokenizer, learning_rate

    elif name == "bart":
        bart_tokenizer = BartTokenizer.from_pretrained(
            "facebook/bart-base", model_max_length=model_length, return_tensors="pt")
        bart_model = BartForSequenceClassification.from_pretrained(
            "facebook/bart-base", num_labels=2)
        learning_rate = 5e-5
        return bart_model, bart_tokenizer, learning_rate

    elif name == "robertatwitter":
        roberta_twitter_tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base", model_max_length=model_length, return_tensors="pt")
        roberta_twitter_model = AutoModelForSequenceClassification.from_pretrained(
            'cardiffnlp/twitter-roberta-base', num_labels=2)
        learning_rate = 5e-5
        return roberta_twitter_model, roberta_twitter_tokenizer, learning_rate

    elif name == "gpt2":
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2", model_max_length=model_length, return_tensors="pt")
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model = GPT2ForSequenceClassification.from_pretrained(
            'gpt2', num_labels=2)
        gpt2_model.config.pad_token_id = gpt2_tokenizer.pad_token_id
        learning_rate = 5e-5
        return gpt2_model, gpt2_tokenizer, learning_rate

    elif name == "electra":
        electra_tokenizer = ElectraTokenizer.from_pretrained(
            'google/electra-base-discriminator', model_max_length=model_length, return_tensors="pt")
        electra_model = ElectraForSequenceClassification.from_pretrained(
            'google/electra-base-discriminator', num_labels=2)
        learning_rate = 5e-5
        return electra_model, electra_tokenizer, learning_rate
        
    elif name == "roberta":
        roberta_tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", model_max_length=model_length, return_tensors="pt", use_fast=False)
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base', num_labels=2)
        learning_rate = 5e-5
        return roberta_model, roberta_tokenizer, learning_rate
    else:
        print('Model not found')
    raise ValueError

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

def collate_fn(batch):
    input_ids = [i["input_ids"] for i in batch]
    attention_mask = [i["attention_mask"] for i in batch]
    token_type_ids = [i["token_type_ids"] for i in batch]
    label = [i["label"] for i in batch]
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "token_type_ids": torch.tensor(token_type_ids),
        "label": torch.tensor(label)
    }


def train(model, tokenized_datasets, args):
    # Split the dataset into training and evaluation sets
    train_ratio = 0.8
    train_size = int(train_ratio * len(tokenized_datasets))
    test_size = len(tokenized_datasets) - train_size
    train_dataset, test_dataset = random_split(tokenized_datasets, [train_size, test_size])
    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    # optimization
    num_epochs = args.num_epochs
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=num_epochs * len(train_dataloader)
    )

    device = args.device
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            ce_loss = loss(outputs.logits, labels)
            ce_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        all_predictions = []
        all_labels = []
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            _, predicted = torch.max(outputs.logits, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_predictions))


def inference(model, val_tokenized_data, args):
    device = args.device
    model.to(device)
    model.eval()
    all_predictions = []
    all_labels = []
    for batch in tqdm(val_tokenized_data, desc="Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        _, predicted = torch.max(outputs.logits, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_predictions))
    return all_predictions, all_labels


if __name__ == '__main__':
    # add argument parser
    parser = argparse.ArgumentParser(description='Training script for shallow models')
    parser.add_argument('--device', type=str, default='cuda:4', help='Device to use for training')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--model_name', type=str, default='convbert', help='Name of the pre-trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the input sequence')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train the model')
    parser.add_argument('--is_train', type=bool, default=True, help='Wheaining Epoch 1/3:ther to train the model or not')
    args = parser.parse_args()

    # use any of the following config names as a key:
    assert args.task_name in ["gender_bias", "hate_speech", "linguistic_bias", "political_bias", "racial_bias", "text_level_bias"]
    # load model, tokenizer, and learning rate
    model, tokenizer, learning_rate = modelspecifications(args.model_name, args.max_length)

    if args.is_train:
        # load dataset
        dataset_dict = load_dataset("mediabiasgroup/mbib-base")
        dataset = dataset_dict[args.task_name]
        print(f"Task Name: {args.task_name}")
        print(f"Number of samples in the dataset: {len(dataset)}")
        print(f"Example data: {dataset[0]}")
        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        train(model, tokenized_datasets, args)
    else:
        # load validation dataset
        dataset_dict = load_dataset("mediabiasgroup/mbib-base")
        dataset = dataset_dict[args.task_name]
        print(f"Task Name: {args.task_name}")
        print(f"Number of samples in the dataset: {len(dataset)}")
        print(f"Example data: {dataset[0]}")
        eval_datasets = dataset.map(preprocess_function, batched=True)
        inference(model, eval_datasets, args)
