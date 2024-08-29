import pandas as pd
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import chi2_contingency


# TODO: holistic view - statistics
# TODO: within domain, how are bias tyoes correlated?
# TODO: across domain, how are the bias types stats differ?
# TODO: across domain, how are bias correlations differ?
# TODO: time series analysis
# TODO: different platforms (if have time)


# for d in domains:
#     df = pd.read_csv(f'predicted-youtube-{d}.csv') #     df['publishedAt'] = pd.to_datetime(df['publishedAt'])
#     bias_preds = df[targets].copy()
#     bias_preds = bias_preds.dropna(subset=['publishedAt'])

#     bias_preds.index = pd.to_datetime(bias_preds.index)
#     # filtered_bias_preds = bias_preds[:'2023-12-31']
#     monthly_counts = bias_preds.groupby(pd.Grouper(key='publishedAt', freq='ME')).sum() 


#     # Plot the monthly trends
#     fig, axes = plt.subplots(3, 1, figsize=(12, 8))
#     y_limits = [(0, 60), (0, 60), (0, 60)]  # Example limits for each subplot

#     for i, bias_type in enumerate(['hateSpeech_pred', 'genderBias_pred', 'politicalBias_pred']):
#         monthly_counts[bias_type].plot(ax=axes[i])
#         axes[i].set_title(f'Monthly Trend of {bias_type}')
#         axes[i].set_xlabel('Month')
#         axes[i].set_ylabel('Count')

#         # Setting hard-coded y-axis limits
#         axes[i].set_ylim(y_limits[i])

#     # Setting x-axis limits to show dates up to December 31, 2023
#     for ax in axes:
#         ax.set_xlim([pd.Timestamp('2021-01-01'), pd.Timestamp('2023-12-31')])

#     plt.suptitle(f'Time Series Analysis of Binary Variables in {d.capitalize()} Domain')
#     plt.tight_layout()
#     plt.savefig(f'./images/monthly_trends_{d}.png')
#     plt.close(fig)

#     correlation_matrix = bias_preds.corr()
#     print("Correlation Matrix:")
#     print(correlation_matrix)

#     contingency_table = pd.crosstab(bias_preds['hateSpeech_pred'], bias_preds['genderBias_pred'])
#     print("Contingency Table (hateSpeech_pred vs genderBias_pred):")
#     print(contingency_table)

#     contingency_table = pd.crosstab(bias_preds['genderBias_pred'], bias_preds['politicalBias_pred'])
#     print("Contingency Table (genderBias_pred vs politicalBias_pred):")
#     print(contingency_table)

#     stat, p, dof, expected = chi2_contingency(contingency_table)
#     print(f"Chi-Square Statistic: {stat:.2f}")
#     print(f"p-value: {p:.4f}")

#     p_a_given_b = contingency_table.loc[1, 1] / contingency_table.loc[:, 1].sum()
#     p_b_given_a = contingency_table.loc[1, 1] / contingency_table.loc[1, :].sum()
#     print(f"P(hateSpeech_pred|genderBias_pred): {p_a_given_b:.2f}")
#     print(f"P(genderBias_pred|hateSpeech_pred): {p_b_given_a:.2f}")

#     bias_preds['hateSpeech_pred'].plot()
#     bias_preds['genderBias_pred'].plot()
#     bias_preds['politicalBias_pred'].plot()
#     plt.legend(['hateSpeech_pred', 'genderBias_pred', 'politicalBias_pred'])
#     plt.title('Time Series Plots of hateSpeech_pred, genderBias_pred, and politicalBias_pred')
#     plt.savefig(f'./images/time_series(GB, PB)-{d}.png')


# gender bias, racial bias,hate speech, linguistic bias, text-level content bias, political bias
domains = ['healthcare', 'politics', 'sports', 'jobEducation', 'entertainment']
bias_mapping = {i: value for i, value in enumerate(["genderBias", "racialBias", 
                "hateSpeech", "linguisticBias",
                "Text-levelContextBias", "politicalBias"])}
platform = 'youtube'
for d in domains:
    BERT_prediction = pd.read_csv(f'shallow_predictions/predicted-{platform}-{d}.csv')
    GPT_prediction = pd.read_json(f'GPT_predictions/{platform}_{d}_GPT_predictions.json', ).transpose()
    GPT_prediction.rename(columns=bias_mapping, inplace=True)
    print(BERT_prediction.head())


