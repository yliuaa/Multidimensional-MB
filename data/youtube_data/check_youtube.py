import pandas as pd
import numpy as np
import json
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report


# biases = ['genderBias', 'racialBias', 'hateSpeech', 'linguisticBias', 'textLevelContentBias', 'politicalBias', 'cognitiveBias']
biases = ['genderBias', 'racialBias',
          'hateSpeech', 'linguisticBias',
          'textLevelContentBias', 'politicalBias']
df_1 = pd.read_csv('youtube(healthcare, politics, sports-100))_Yifan.csv')
df_1_added = pd.read_csv('youtube(entertainment-job-100)) -Yifan.csv')
df_2 = pd.read_csv('youtube(healthcare, politics, sports-100))_Yike.csv')
df_2_added = pd.read_csv('youtube(entertainment-job-100)) - Yike.csv')

df_1 = pd.concat([df_1, df_1_added], axis=0).reset_index(drop=True)
df_2 = pd.concat([df_2, df_2_added], axis=0).reset_index(drop=True)

overlapped_ids = list(range(41, 61)) + list(range(141, 161)) + list(range(241, 261)) + list(range(341, 361)) + list(range(441, 461))

print("Inter-rater Agreement")
for bias in biases:
    annotation_1 = df_1.iloc[overlapped_ids][bias]
    annotation_2 = df_2.iloc[overlapped_ids][bias]
    print(bias)
    print(cohen_kappa_score(annotation_1, annotation_2))



print(df_1.loc[1, 'hateSpeech'])
data = []
for i in range(len(df_1)):
    if not np.isnan(df_1.loc[i, 'hateSpeech']):
        data.append(df_1.iloc[i])
    else:
        data.append(df_2.iloc[i])
new_df = pd.DataFrame(data, index=None)


pred_type='GPT'
domains = ['politics', 'sports', 'healthcare', 'jobEducation', 'entertainment']
if pred_type == 'GPT':
    all_domain = []
    for d in domains:
        with open(f'../youtube_{d}_GPT_predictions.json', "r") as read_file:
            predictions = json.load(read_file)
            pred_df = pd.DataFrame({
                'id': predictions.keys(),
                'predictions': predictions.values()
            })
            all_domain.append(pred_df)

else:
    all_domain = []
    for domain in domains:
        pred_df = pd.read_csv(f'../predicted-youtube-{domain}.csv')
        all_domain.append(pred_df)

    pred_biases = ['genderBias_pred', 'racialBias_pred',
                   'hateSpeech_pred', 'linguisticBias_pred',
                   'textLevelBias_pred', 'politicalBias_pred']

stacked_df = pd.concat(all_domain, axis=0)
stacked_df = stacked_df.reset_index(drop=True)
if pred_type != 'GPT':
    stacked_df['predictions'] = np.array(stacked_df[pred_biases].to_numpy()).tolist()
pred_df = stacked_df


new_df = new_df.drop_duplicates(subset='id', keep='first')
pred_df = pred_df.drop_duplicates(subset='id', keep='first')
for i, domain in enumerate(domains):
    val_gt = new_df[new_df['domain'] == domain]
    val_gt = val_gt[val_gt['id'].isin(pred_df['id'])].sort_values(by='id')
    val_pred = pred_df[pred_df['id'].isin(val_gt['id'])].sort_values(by='id')
    pred = val_pred['predictions'].apply(lambda x: [int(i) for i in x])
    pred_arr = np.array(pred.tolist())
    print(f"============= ACC-evaluation({domain}) ============")
    for i, bias in enumerate(biases): 
        target_pred = pred_arr[:, i]
        print(classification_report(target_pred, val_gt[bias]))
        wrong_predictions = [i for i, (true, pred) in enumerate(zip(target_pred, val_gt[bias])) if true != pred]
        print(f"bias type({bias}): {len(wrong_predictions)}")
        val_pred.iloc[wrong_predictions].to_csv(f'wrong_prediction{(bias)}.csv')