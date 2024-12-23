import asyncio
import re
import time
import json
import sys
import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import date
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext, models
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from loguru import logger
import argparse


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


@dataclass
class Context:
    HateSpeech: str
    LinguisticBias: str
    TextLevelContextBias: str
    PoliticalBias: str
    RacialBias: str
    GenderBias: str


def load_data(path):
    df = pd.read_csv(path)

    logger.info(f"Loading data from {path}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Shape: {df.shape}")
    return df


# def confidence_grounding():

#     return


class Pipeline:
    def __init__(
        self,
        model,
        max_retries,
        eval_data,
        batch_size,
        example_path,
        output_path,
        strategies=["zero-shot", "CoT"],
        dynamic_context=True,
    ):
        self.model = model
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.example_path = example_path
        self.output_path = output_path
        self.max_retries = max_retries
        self.task_experts = []
        self.aggregator = None
        self.contextual_expert = None

        self.examples = json.load(open(example_path, "r"))
        self.reflection_bank = None
        self.strtegies = strategies
        self.dynamic_context = dynamic_context
        self.conf_check = True
        self.saved_ids = []

        logger.debug(f"Confidence check {self.conf_check}")
        # initialize the agent
        self._init_agent()

    def _base_prompt(self):
        # 1) Hate Speech: Hate speech refers to any language that manifests hatred towards a specific group or aims to degrade, humiliate, or offend.
        return (
            f"""
            # Task: Multidimensional Media Bias Annotation
            You are a media bias expert, you will output binary judgement on whether a media bias ({bias_dimensions}) is shown in the input text. The media bias dimensions are defined as follows:

            # Definitions:
            1) Hate Speech: expressions or statements that contain offensive, aggresive or derogatory language towards individuals or groups based on attributes like race religion, gender, or other identity characteristics.
            2) Linguistic Bias: is characterized by biases stemming from lexical features, manifesting through the **specific choices of words** that reflect social-category cognition attributed to described groups or individuals.
            3) Text-level Context Bias: is characterized by the strategic use of specific words and statements within texts to **craft a narrative that favors a particular viewpoint** of an event. This includes presenting a skewed description of incidents, thereby biasing the narrative towards one side of an argument.
            4) Political Bias: refers to the expression of a particular political ideology, altering the political discourse by emphasizing certain viewpoints over others, thereby shaping the agenda of political discussions and debates.
            5) Racial Bias: is defined as texts that express negative or positive descriptions towards racial groups, which could play a pivotal role in shaping public discourse and policy-makingrelated to race relations and diversity.
            6) Gender Bias: Gender **underrepresentation, reliance on stereotypes, or negative portrayals**—each of which contributes to the perpetuation.

            # Confidence grounding:
            To ensure the quality of your predictions, consider the following confidence levels:
            - 0.0: No confidence
            - 0.2: Low confidence
            - 0.5: Medium confidence
            - 0.8: High confidence
            - 1.0: Absolute confidence

            """,
            """# Output format:
            Your output type will be Prediction which contains a **list** of binary integer predictions and a **list** of float confidences, based on your expert opinion, where 1 indicates such bias exists. **No other output type is accepted! Double check your format.**

            # Example:
            Prediction(Prediction=[0, 0, 0, 0, 0, 0], Confidence=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
            """,
        )

    def _init_agent(self):
        strategies = self.strtegies
        task_prompt, format_prompt = self._base_prompt()
        for strategy in strategies:
            if strategy == "CoT":
                self.task_experts.append(
                    (
                        strategy,
                        Agent(
                            self.model,
                            result_type=Prediction,
                            system_prompt=(
                                task_prompt
                                + """
                    # Strategy: 
                    To ensure a thorough analysis, follow these steps for every bias dimension:

                    1. Identify Cues: Look for specific words, phrases, or context that align with the bias definition.
                    2. Evaluate Context: Assess whether the identified cues explicitly or implicitly exhibit the given type of bias.
                    3. Weigh Evidence: Consider the presence and intensity of bias-related language or framing.
                    4. Make a Judgment: Based on the evidence, output a binary prediction (1 for biased, 0 for not biased).
                    5. Assess Confidence: Reflect on your judgment and assign a confidence score (a float in the range (0, 1)) for each prediction.

                    """
                                + format_prompt,
                            ),
                            retries=self.max_retries,
                        ),
                    )
                )

            if strategy == "CoT-backward":
                self.task_experts.append(
                    (
                        strategy,
                        Agent(
                            self.model,
                            result_type=Prediction,
                            system_prompt=(
                                task_prompt
                                + """
                    # Strategy: 
                    To ensure a thorough analysis using a conclusion-first approach, follow these steps for every bias dimension:

                    1. Hypothesize a Conclusion: Start by assuming a binary classification (1 for biased, 0 for not biased) as the initial hypothesis.
                    2. Trace Supporting Evidence: Identify specific words, phrases, or context that support the hypothesized conclusion.
                    3. Challenge the Hypothesis: Search for counter-evidence or alternative explanations that could disprove the hypothesis.
                    4. Validate or Revise: Based on the supporting and counter-evidence, confirm or adjust the initial hypothesis to finalize the judgment.
                    5. Assess Confidence: Reflect on the strength and reliability of the evidence and assign a confidence score (a float in the range (0, 1)) for the final prediction.
                    """
                                + format_prompt,
                            ),
                            retries=self.max_retries,
                        ),
                    )
                )
            elif strategy == "zero-shot":
                self.task_experts.append(
                    (
                        strategy,
                        Agent(
                            self.model,
                            result_type=Prediction,
                            system_prompt=(task_prompt + format_prompt),
                            retries=self.max_retries,
                        ),
                    )
                )

        self.aggregator = Agent(
            self.model,
            result_type=Prediction,
            system_prompt=(
                task_prompt
                + f"""
            # Strategy:

            You are given a social media post and annotations from multiple LLM predictions. Your task is to make your own judgement with reference to these predictions with reference to the context provided. You should not rely on the predictions alone, but rather use them as a reference to make your own judgement. The context is provided to help you make a more informed decision. Judge whether the text is biased in any of the dimensions ({bias_dimensions}) following the order provided. For each dimension, make a binary prediction and assign a confidence score to your prediction.
            """
                + format_prompt
            ),
            retries=self.max_retries,
        )
        self.contextual_expert = Agent(
            self.model,
            result_type=Context,
            system_prompt=(
                """
                # Task: Reflection on Mistake Annotations
                You are a reflective annotation expert tasked with analyzing mistake annotations and generating useful contextual information for a multidimensional bias annotation expert.

                # Definitions:
                1) Hate Speech: expressions or statements that contain offensive, aggresive or derogatory language towards individuals or groups based on attributes like race religion, gender, or other identity characteristics.
                2) Linguistic Bias: is characterized by biases stemming from lexical features, manifesting through the **strong vulgar words** that reflect social-category cognition attributed to described groups or individuals.
                3) Text-level Context Bias: is characterized by the strategic use of specific words and statements within texts to **craft a narrative that favors a particular viewpoint** of an event. This includes presenting a skewed description of incidents, thereby biasing the narrative towards one side of an argument.
                4) Political Bias: refers to the expression of a particular political ideology, altering the political discourse by emphasizing certain viewpoints over others, thereby shaping the agenda of political discussions and debates.
                5) Racial Bias: is defined as texts that express negative or positive descriptions towards racial groups, which could play a pivotal role in shaping public discourse and policy-makingrelated to race relations and diversity.
                6) Gender Bias: Gender **underrepresentation, reliance on stereotypes, or negative portrayals**—each of which contributes to the perpetuation.
                
                # Instructions:
                1. Analyze the Mistake Annotations: For each mistake annotation, identify the specific error made in the annotation process (e.g., misclassification, overlooked context, ambiguity in input). Reflect on why the mistake occurred and what could have prevented it.
                2. Refer to Positive Examples: If necessary, consult the pre-prepared positive examples to clarify ground truth standards. When referring to these examples, add "Examples: {examples}" to the context string. Highlight the postive examples should be assigned with absolute confidence.
                3. Based on your reflection, extract key **generalized** insights irrespective of the specific examples. Ensure the contextual string is clear and can be easily used as a guideline for improving future predictions.

                **For bias dimension without mistakes, output empty string. Double check your format, no other formats are allowed.**

                # Output Format:
                Context(HateSpeech="Example: {example} {reflection}", LinguisticBias="Example: {example} {reflection} ...", TextLevelContextBias="Example: {example} {reflection} ...", PoliticalBias="Example: {example} {reflection} ...", RacialBias="Example: {example} {reflection} ...", GenderBias="Example: {example} {reflection} ...")

                """
            ),
            retries=20,
        )

    async def prompt_ens(self, text):
        res = {}

        async def run_expert(strategy, expert):
            return strategy, await expert.run(text)

        tasks = [run_expert(strategy, expert) for strategy, expert in self.task_experts]
        outs = await asyncio.gather(*tasks)
        res = {strategy: expert_result.data for strategy, expert_result in outs}
        return res

    async def confidence_aware_agg(
        self, outs, context, label, index, conf_check=True, train=True
    ):
        if train:
            assert label is not None
        if conf_check:
            expert_outs = outs.values()
            conf = np.asarray([pred.Confidence for pred in expert_outs])
            preds = np.asarray([pred.Prediction for pred in expert_outs])

            pos_indices = np.where(np.any(preds == 1, axis=0))[0]
            if pos_indices.size > 0:
                pos_conf = conf[:, np.where(np.any(preds == 1, axis=0))[0]]
                max_conf_gap = np.max(
                    np.max(pos_conf, axis=0) - np.min(pos_conf, axis=0)
                )
                logger.debug(f"Maximum confidence gap: {max_conf_gap}")
                if max_conf_gap > 0.2:
                    return self.human_judgement(index, label, train)

        res = await self.aggregator.run(str(outs), deps=context)
        return res.data

    def human_judgement(self, index, label, train):
        self.saved_ids.append(index)
        if train:
            return Prediction(
                Prediction=[int(label[i]) for i in range(len(label))],
                Confidence=[1.0 for _ in range(len(label))],
            )
        else:
            return Prediction(
                Prediction=[-1 for _ in range(len(label))],
                Confidence=[1.0 for _ in range(len(label))],
            )

    @logger.catch
    async def annotate(self, sample_num, train=True, note=""):

        if sample_num != -1:
            test_dataset = self.eval_data.sample(n=sample_num)
        else:
            test_dataset = self.eval_data
            logger.info(f"Using subset size: {sample_num}")
        texts = test_dataset["textDisplay"].to_numpy()
        ids = test_dataset["id"].to_numpy()
        gt_annotations = test_dataset[bias_dimensions].to_numpy()
        ref_examples = json.load(open(self.example_path, "r"))
        logger.info("Loaded context examples at " + self.example_path)
        logger.info("Starting annotation...")

        predictions = np.empty((0, len(bias_dimensions)), int)
        indices = np.empty((0,), int)
        annotated_texts = np.empty((0,), str)
        labels = np.empty((0, len(bias_dimensions)), int)
        context = ""
        for start in tqdm(range(0, len(ids), self.batch_size), desc=f"Annotating"):
            end = start + self.batch_size
            batch_texts = texts[start:end]
            batch_indices = ids[start:end]
            batched_labels = gt_annotations[start:end, :]
            batched_preds = []
            local_retries = 0
            while local_retries < self.max_retries:
                try:
                    for i, text in enumerate(batch_texts):
                        res_ens = await self.prompt_ens(text)
                        if len(self.task_experts) == 1:
                            res = res_ens[self.task_experts[0][0]]
                        else:
                            res = await self.confidence_aware_agg(
                                res_ens,
                                context,
                                batched_labels[i],
                                batch_indices[i],
                                conf_check=self.conf_check,
                                train=train,
                            )
                        batched_preds.append(res)
                        print(res)
                    break
                except Exception as e:
                    local_retries += 1
                    logger.warning(
                        f"Batch failed - retrying ({local_retries}/{self.max_retries}) due to error: {e}"
                    )
                    # time.sleep(10)
            else:
                logger.error(
                    f"Batch failed after {self.max_retries} retries - skipping batch"
                )
                continue

            # Discard id if there is -1 in the prediction
            discard_ids = [
                i for i, pred in enumerate(batched_preds) if -1 in pred.Prediction
            ]
            if discard_ids:
                logger.warning(f"Discarding {len(discard_ids)} samples")
                batched_preds = [
                    pred for i, pred in enumerate(batched_preds) if i not in discard_ids
                ]
                batch_indices = [
                    idx for i, idx in enumerate(batch_indices) if i not in discard_ids
                ]
                batch_texts = [
                    text for i, text in enumerate(batch_texts) if i not in discard_ids
                ]
                batched_labels = [
                    label
                    for i, label in enumerate(batched_labels)
                    if i not in discard_ids
                ]

            batched_preds = np.asarray(
                [pred.Prediction for pred in batched_preds]
            ).astype(np.int16)
            predictions = np.concatenate((predictions, batched_preds))
            indices = np.concatenate((indices, np.asarray(batch_indices)))
            annotated_texts = np.concatenate((annotated_texts, np.asarray(batch_texts)))
            labels = np.concatenate((labels, np.asarray(batched_labels))).astype(
                np.int16
            )

            logger.info(f"Batch annotated - idx {start}:{end}")
            all_labels = labels
            all_preds = predictions

            if (start // self.batch_size) % 10 == 0 and start != 0 and train:
                while local_retries < self.max_retries:
                    try:
                        context, _ = await self.evaluate(
                            all_preds, all_labels, annotated_texts, ref_examples
                        )
                        break
                    except Exception as e:
                        local_retries += 1
                        logger.warning(
                            f"Reflection failed - retrying ({local_retries}/{self.max_retries}) due to error: {e}"
                        )

        # ------------------------- Test and Save Predictions ------------------------ #
        mistakes = {}
        if self.output_path is not None:
            _, mistakes = await self.evaluate(
                all_preds, all_labels, annotated_texts, ref_examples
            )
            final_annotations = {
                "ids": np.asarray(indices),
                "texts": np.asarray(annotated_texts),
                "predictions": all_preds,
            }
            np.savez(
                self.output_path + f"{note}_predictions.npz", data=final_annotations
            )
            np.savez(self.output_path + f"{note}_mistakes.npz", data=mistakes)
            logger.success(
                "Pipeline finished - saved predictions at " + self.output_path
            )
        logger.success("Done.")

    @logger.catch
    async def evaluate(self, all_preds, all_labels, texts, examples):
        saved_number = len(self.saved_ids)
        logger.info(
            f"Evaluation started; current human judgement({saved_number} - {saved_number/len(all_labels)}): {self.saved_ids}"
        )
        context, reflection = "", ""
        mistakes = {}

        for idx, bias in enumerate(bias_dimensions):
            bias_preds = all_preds[:, idx]
            bias_labels = all_labels[:, idx]
            bias_reports = classification_report(
                bias_labels, bias_preds, output_dict=True
            )
            wrong_ids = np.where((bias_labels != bias_preds))[0]
            if texts.size != 0 and wrong_ids.size > 0:
                wrong_ids_list = wrong_ids.tolist()
                mistake_texts = texts[wrong_ids_list]
                wrong_preds = bias_preds[wrong_ids_list]
                bias_mistakes = {"texts": mistake_texts, "predictions": wrong_preds}
                mistakes[bias] = bias_mistakes

            acc = bias_reports["accuracy"]
            macro_f1 = bias_reports["macro avg"]["f1-score"]
            micro_f1 = bias_reports["weighted avg"]["f1-score"]
            logger.info(
                f"{bias}: Macro F1 {macro_f1}; Micro F1 {micro_f1}; Accuracy {acc}\n"
            )

        if self.dynamic_context:
            logger.info("Reflecting on mistakes...")
            reflection = await self.contextual_expert.run(
                str(mistakes),
                deps=f"Examples: {examples}",
                message_history=self.reflection_bank,
            )
            self.reflection_bank = reflection.all_messages()
            if reflection != "":
                context = reflection.data
            logger.debug(context)
        logger.success("Evaluation done.")
        print(mistakes)
        return context, mistakes


if __name__ == "__main__":

    # --------------------------------- arguments -------------------------------- #
    parser = argparse.ArgumentParser(
        description="Annotate text data with expert models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="The model to use for annotation",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="../api_key.json",
        help="The corresponding API key",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero-shot", "CoT", "biliteral"],
        help="The strategies to use for annotation",
    )
    parser.add_argument(
        "--enable_context",
        dest="dynamic_context",
        action="store_true",
        help="Whether to use dynamic context",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="The maximum number of retries for a failed annotation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The seed for the random number generator"
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="~/Multidimensional-MB/data/reddit_data/eval_dataset(Nov5).csv",
        help="The path to the evaluation dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="The batch size for annotation"
    )
    parser.add_argument(
        "--context_path",
        type=str,
        default="./ref_samples.json",
        help="The path to the reference examples",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./save/",
        help="The path to save the predictions",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="Development run",
        help="The note for this annotation run",
    )
    args = parser.parse_args()

    # with open("./config.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    #     args = argparse.Namespace(**config)

    np.random.seed(args.seed)
    with open(args.api_key, "r") as f:
        key = json.load(f)
    logger.add(f"../logs/{date.today()}-{args.note}-{args.strategies}.log", mode="w")
    logger.info("Pipeline started with args: " + str(args))

    # --------------------------------- run pipeline -------------------------------- #
    data = load_data(args.eval_path)
    if "gpt" in args.model:
        model = OpenAIModel(args.model, api_key=key["OPENAI"])
    elif "gemini" in args.model:
        model = GeminiModel(args.model, api_key=key["GOOGLE"])

    pipeline = Pipeline(
        model,
        args.max_retries,
        data,
        args.batch_size,
        args.context_path,
        args.output_path,
        strategies=args.strategies,
        dynamic_context=args.dynamic_context,
    )

    # Use only 1000 samples for evaluation
    asyncio.run(pipeline.annotate(sample_num=-1, note=args.note))
