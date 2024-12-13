import re
import time
import json
import sys
import pandas as pd
import numpy as np
import pandas as pd
from typing import List, Tuple
from swarm import Swarm
from datetime import datetime
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from loguru import logger
import argparse

import yaml

# --------------------------------- constants -------------------------------- #
bias_dimensions = [
    "hate speech",
    "linguistic bias",
    "text-level context bias",
    "political bias",
    "racial bias",
    "gender bias",
]

# ---------------------------------- helpers --------------------------------- #
@dataclass
class Prediction:
    Prediction: List[int]
    Confidence: List[float]

    def __post_init__(self):
        bias_num = len(bias_dimensions)
        if len(self.Prediction) != bias_num:
            raise ValueError(f"Prediction list must have {bias_num} elements")
        if len(self.Confidence) != bias_num:
            raise ValueError(f"Cofidence list must have {bias_num} elements")


def load_data(path):
    return pd.read_csv(path)


class Pipeline:
    def __init__(self, model, max_retries, eval_data, batch_size, context_path, output_path):
        self.model = model
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.context_path = context_path
        self.output_path = output_path
        self.max_retries = max_retries
        self.task_expert = None
        
        # initialize the agent
        self._init_agent()

    def _init_agent(self):
        self.task_expert = Agent(
            self.model,
            result_type=Prediction,
            system_prompt=(
            f"""
            You are a media bias expert, you will output binary judgement on whether a media bias ({bias_dimensions}) is shown in the input text. The media bias dimensions are defined as follows:

            1) Hate Speech: is characterized by language that expresses hatred towards a group or individual(s). On social media, hate speech represents not only a form of biased text but also a significant cultural threat that has been linked to crimes against minorities.
            2) Linguistic Bias: is characterized by biases stemming from lexical features, manifesting through the specific choices of words that reflect social-category cognition attributed to described groups or individuals.
            3) Text-level Context Bias: is characterized by the strategic use of specific words and statements within texts to craft a narrative that favors a particular viewpoint of an event. This includes presenting a skewed description of incidents, thereby biasing the narrative towards one side of an argument.
            4) Political Bias: refers to the expression of a particular political ideology, altering the political discourse by emphasizing certain viewpoints over others, thereby shaping the agenda of political discussions and debates.
            5) Racial Bias: is defined as texts that express negative or positive descriptions towards racial groups, which could play a pivotal role in shaping public discourse and policy-makingrelated to race relations and diversity.
            6) Gender Bias: encompasses discrimination against gender groups as manifested in textual content. This form of discrimination can appear through underrepresentation, reliance on stereotypes, or negative portrayals—each of which contributes to the perpetuation of gender inequality.

            For example, your output type will be Prediction which contains a list of binary predictions and a list of float prediction (for either biased or not biased) confidences, based on your expert opinion, where 1 indicates such bias exists. The prediction should be of type integer, and confidence score should be a float in range (0, 1). No other output type is accepted.
            """
            ),
        )
        self.task_expert._max_result_retries = self.max_retries

        @self.task_expert.system_prompt
        def reference_samples(ctx: RunContext[str]) -> str:
            return f""" These examples represent positive samples of each bias dimensions. Use these as guidance to understand what constitutes a 'Positive' example. In addition, they are can be considered as full confidence examples (1.0).

            {ctx}
            """

        
    def update_context(self, new_context):
        # self.task_expert._system_prompt = new_context
        pass


    def run(self, sample_num, inference_only=False):
        if sample_num != -1:
            test_dataset = self.eval_data.sample(n=sample_num)
        else:
            test_dataset = self.eval_data
        texts = test_dataset["textDisplay"].to_numpy()
        ids = test_dataset["id"].to_numpy()
        gt_annotations = test_dataset[bias_dimensions].to_numpy()
        ref_examples = json.load(open(self.context_path, "r"))
        logger.info("Loaded context examples: " + str(ref_examples))
        logger.info("Starting annotation...")

        indices = []
        predictions = []
        labels = []
        for start in tqdm(range(0, len(ids), self.batch_size), desc=f"Annotating"):
            end = start + self.batch_size
            batch_texts = texts[start:end]
            batch_indices = ids[start:end]
            batched_labels = gt_annotations[start:end, :]
            local_retries = 0
            while local_retries < self.max_retries:
                batched_preds = []
                try:
                    for text in batch_texts:
                        result = self.task_expert.run_sync(text, deps=ref_examples)
                        batched_preds.append(result.data)
                    break
                except Exception as e:
                    local_retries += 1
                    logger.warning(f"Batch failed - retrying ({local_retries}/{self.max_retries}) due to error: {e}")
                    time.sleep(10)
            else:
                logger.error(f"Batch failed after {self.max_retries} retries - skipping batch")
                continue

            predictions.extend(batched_preds)
            indices.extend(batch_indices)
            labels.extend(batched_labels)
            logger.info(f"Batch annotated - starting idx {start}")

            if (start // self.batch_size) % 5 == 0 and start != 0 and not inference_only:
                all_labels = np.stack(labels, axis=0)
                assert all_labels.shape[0] == len(predictions)
                self.evaluate(predictions, all_labels)

                # TODO: add contextual updates here


        # Save predictions
        if self.output_path is not None:
            self.evaluate(predictions, all_labels)
            np.save(self.output_path, predictions, allow_pickle=True)
            np.save(self.output_path + "_predictions.npy", all_labels, allow_pickle=True)
            logger.success("Pipeline finished - saved predictions at " + self.output_path)
        logger.success("Done.")

    @logger.catch
    def evaluate(self, predictions, all_labels):
        logger.info("Evaluating")
        logger.debug(predictions)
        all_preds = np.asarray([pred.Prediction for pred in predictions]).astype(np.int16)
        for idx, bias in enumerate(bias_dimensions):
            bias_preds = all_preds[:, idx]
            bias_labels = all_labels[:, idx]
            bias_reports = classification_report(bias_labels, bias_preds, output_dict=True)

            logger.info(f"Metrics for {bias}: {bias_reports} \n")

        logger.success("Evaluation done.")



# @task_experts.tool
# async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
#     """Check if the square is the winner or loser"""
#     return 'winner' if square == ctx.deps else 'loser'


if __name__ == "__main__":

    # --------------------------------- arguments -------------------------------- #
    parser = argparse.ArgumentParser(description='Annotate text data with expert models')
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='The model to use for annotation')
    parser.add_argument('--api_key', type=str, default="../api_key.txt", help='The corresponding API key')
    parser.add_argument('--max_retries', type=int, default=5, help='The maximum number of retries for a failed annotation')
    parser.add_argument('--seed', type=int, default=42, help='The seed for the random number generator')
    parser.add_argument('--eval_path', type=str, default="/home/yifan/Desktop/tidy/PapersNProjects/Code/MMediaBias/Multidimensional-MB/data/reddit_data/eval_dataset(Nov5).csv", help='The path to the evaluation dataset')
    parser.add_argument('--batch_size', type=int, default=10, help='The batch size for annotation')
    parser.add_argument('--context_path', type=str, default="./ref_samples.json", help='The path to the reference examples')
    parser.add_argument('--output_path', type=str, default="./save/", help='The path to save the predictions')
    parser.add_argument('--note', type=str, default="Development run", help='The note for this annotation run')
    args = parser.parse_args()

    # with open("./config.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    #     args = argparse.Namespace(**config)


    np.random.seed(args.seed)
    key = open(args.api_key, "r").read()
    logger.add(f"../logs/DevRun-{args.note}.log", mode="w")
    logger.info("Pipeline started with args: " + str(args))

    # --------------------------------- run pipeline -------------------------------- #
    data = load_data(args.eval_path)
    model = OpenAIModel(args.model, api_key=key)
    pipeline = Pipeline(model, args.max_retries, data, args.batch_size, args.context_path, args.output_path)
    pipeline.run(sample_num=1000)