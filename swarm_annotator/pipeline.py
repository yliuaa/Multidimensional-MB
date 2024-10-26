import torch
import torch.nn as nn
import pandas as pd
from bias_experts import TaskExpert, Aggregation
from swarm import Swarm, Agent


# def create_expert(task):
#     return


if __name__ == "__main__":

    # # Test for political bias task
    # print("Start annotating...")
    # client = Swarm()
    # prompt_config = {"prompt_setting": "zero-shot"}
    # gender_bias_expert = TaskExpert(client, None, None, prompt_config, "Racial Bias")

    # # Racial Bias example
    # gender_bias_expert.forward("Trump is the best.")
    # gender_bias_expert.forward("Women should be in kitchen.")
    # gender_bias_expert.forward("Black people should go pick cotton.")
    # gender_bias_expert.forward("Typical Asians. They are good at math.")

    # # Test aggregation
    # aggregator = Aggregation("llm")
    # example_experts_outputs = {
    #     "raw input": "Asians make the best doctors. It's their math skills.",
    #     "expert predictions (prediction, confidence)": [
    #         {"RoBERTa-Twitter": (0, 0.9)},
    #         {"few-shot GPT4o": (0, 0.6)},
    #         {"zero-shot GPT4o": (1, 0.8)},
    #         {"CoT GPT4o": (1, 0.3)},
    #     ],
    # }

    # aggregator.agg(example_experts_outputs)

    # Read data
    input_data = pd.read_excel("example.xlsx")
    print(input_data.head())
    print(list(input_data.columns))
