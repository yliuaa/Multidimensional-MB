import time
import ast
import re
import json
from tqdm.auto import tqdm

from openai import OpenAI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# NOTE: replace secret key with your own.
secret_key = None
client = OpenAI(max_retries=5, api_key=secret_key)
biases = ['genderBias', 'racialBias', 'hateSpeech', 'linguisticBias', 'politicalBias']
df_1 = pd.read_csv('./annotations/youtube(healthcare, politics, sports-100))_annotator1.csv')
df_2 = pd.read_csv('./annotations/youtube(healthcare, politics, sports-100))_annotator2.csv')
data = []
for i in range(len(df_1)):
    if not np.isnan(df_1.loc[i, 'hateSpeech']):
        data.append(df_1.iloc[i])
    else:
        data.append(df_2.iloc[i])
annotations = pd.DataFrame(data, index=None)


# 1. embedding and randomforest
def get_embedding(text: str, model="text-embedding-3-small", **kwargs):
    # replace newlines, which can negatively affect performance.
    if type(text) != str:
        print("Abnormal inputs detected, converting to string: " + str(text))
        text = str(text)

    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    return response.data[0].embedding


def fetch_embeddings(target='racialBias', mode='train'):
    # embedding_model = "text-embedding-3-small"
    # embedding_encoding = "cl100k_base"
    # max_tokens = 8000  # the maximum for text-embedding-3-small is 8191
    # annotations['annotated_embedding'] = annotations.textDisplay.apply(lambda x: get_embedding(x))
    # annotations.to_csv('./annotations/openai_embeddings.csv')

    # target = 'racialBias'
    train_path = '/home/yifan40/bias_identification/mbib-aggregated/racial-bias.csv'
    train_set = pd.read_csv(train_path)

    train_set['text_embedding'] = train_set.text.apply(lambda x: get_embedding(x))
    train_set.to_csv('./mbib-aggregated/racial-bias-embeddings.csv')



# Example function to get the length of each embedding
def get_embedding_length(embedding):
    if isinstance(embedding, str):
        # Convert string representation to list
        embedding = eval(embedding)
    return len(embedding)

def train_randomforest(data_path):
    train_set = pd.read_csv(data_path)

    train_set.dropna(subset=['text_embedding', 'label'], inplace=True)
    train_set['text_embedding'] = train_set['text_embedding'].astype(str)
    train_set['label'] = train_set['label'].astype(str)
    # indices_to_drop = train_set[train_set['text_embedding'].apply(get_embedding_length) < 300].index

    # train_set.drop(indices_to_drop, inplace=True)
    print(train_set.head())

    X = train_set.text_embedding.apply(ast.literal_eval).apply(np.array)
    y = train_set.label.apply(ast.literal_eval).apply(np.array)
    # X_truncated = X.apply(lambda x: np.array(x)[:8000])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = RandomForestClassifier(max_depth=2)
    clf.fit(X_train, y_train)
    # Eval scores
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


# GPT 3.5 prompting based approach, zero-shot
def prompt_based(data_path, domain):
    data = pd.read_csv(data_path)
    predictions = []
    ids = []
    print(len(data))
    for i in tqdm(range(len(data))):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                    "role": "system",
                    "content": "You are a media bias expert. You will detect the media biases (gender bias, racial bias,hate speech, linguistic bias, text-level content bias, political bias) and provide binary predictions in python list format (e.g., [0, 0, 0, 0, 0, 0])"
                    },
                    {
                    "role": "user",
                    "content": data.loc[i, 'textDisplay']
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        
            list_str = re.search(r'\[.*?\]', response.choices[0].message.content).group()
            ls = ast.literal_eval(list_str)
            ls += ['0'] * (6 - len(list_str))
            predictions.append(ls)
            ids.append(data.loc[i, 'id'])
        except Exception as e:
            print(e)
            time.sleep(5)

    predictions = np.asarray(predictions)
    results = {id_val: predictions[i].tolist() for i, id_val in enumerate(ids)}
    # np.save('./prediction.npy', predictions)
    # Saving the dictionary as JSON
    file_path = f'./youtube_{domain}_GPT_predictions.json'
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":

    # Train random forest
    # fetch_embeddings()
    # train_randomforest('./mbib-aggregated/racial-bias-embeddings.csv')
    domains = ['politics', 'sports']
    for d in domains:
        prompt_based(f'./annotations/youtube_{d}.csv', d)

