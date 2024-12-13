import torch
import torch.nn as nn
import pandas as pd
from utils import *
from tqdm import tqdm
from swarm import Swarm, Agent


class BiasIdenSwarm:
    def __init__(
        self,
        api_client,
        expert_configs,
        aggregation_strategy,
        bias_dimensions,
        eval_dataset,
        note=None,
    ):
        self.bias_dimensions = bias_dimensions
        self.bias_swarm = {
            bias: BiasExperts(api_client, expert_configs, bias)
            for bias in bias_dimensions
        }
        assert len(self.bias_swarm) == len(bias_dimensions)
        self.aggregator = Aggregation(api_client, aggregation_strategy)
        self.eval_dataset = eval_dataset
        self.note = note

    def _test_components(self):
        example_experts_outputs = {
            "raw input": "Asians make the best doctors. It's their math skills.",
            "expert predictions (prediction, confidence)": [
                {"RoBERTa-Twitter": (0, 0.9)},
                {"few-shot GPT4o": (0, 0.6)},
                {"zero-shot GPT4o": (1, 0.8)},
                {"CoT GPT4o": (1, 0.3)},
            ],
        }
        test_experts = self.bias_swarm["racial bias"].forward(
            example_experts_outputs["raw input"]
        )
        test_agg = self.aggregator.agg(example_experts_outputs)
        print(" --- Testing with racial bias --- ")
        print("Testing experts: ", test_experts)
        print("Testing aggregation: ", test_agg)

    def annotate_eval(self, texts):
        total_predictions = {bias: [] for bias in self.bias_dimensions}
        uni_predictions = {bias: [] for bias in self.bias_dimensions}
        # texts = self.eval_dataset["textDisplay"]
        for idx, text in enumerate(texts):
            print("finished ", idx)
            for bias in self.bias_dimensions:
                experts_out = self.bias_swarm[bias].forward(text)
                uni_results = experts_out["expert_predictions (prediction, confidence)"]
                out = self.aggregator.agg(experts_out)
                # Parse the output "{final label} {reasoning}", make it a dictionary
                final_label_str, reasoning = out.split(" ", 1)
                final_label_str = final_label_str.replace("'", "").replace('"', "")
                json_out = {"Final label": final_label_str, "Reasoning": reasoning}
                total_predictions[bias].append(int(json_out["Final label"]))
                uni_predictions[bias].append(uni_results)

        return total_predictions, uni_predictions


class BiasExperts(nn.Module):
    def __init__(self, client, expert_configs, task, verbose=False):
        super().__init__()
        self.task = task
        self.verbose = verbose
        self.client = client
        self.expert_configs = expert_configs
        if client == None:
            raise ValueError("OpenAI client not defined.")
        self.experts = self._initialize_experts(task)

    def _get_task(self):
        return self.task

    def _initialize_experts(self, task):
        if self.expert_configs is None:
            raise ValueError("Expert configurations not defined.")

        experts = {}
        for config in self.expert_configs:
            experts[config["strategy"]] = TaskExpert(self.client, config, task)
        # print("Initialized experts: ", self.expert_configs)
        return experts

    def forward(self, text):
        experts_out = {
            "raw_input": text,
            "expert_predictions (prediction, confidence)": [],
        }

        for key, expert in self.experts.items():
            response = expert.forward(text)
            experts_out["expert_predictions (prediction, confidence)"].append(
                {str(key): str(response)}
            )
        if self.verbose:
            print(f"Responses for {self.task}: ", experts_out)
        return experts_out

        # for idx, text in enumerate(texts):
        #     experts_out = {
        #         "raw_input": text,
        #         "expert_predictions (prediction, confidence)": [],
        #     }

        #     for key, expert in self.experts.items():
        #         response = expert.forward(text)
        #         experts_out["expert_predictions (prediction, confidence)"].append(
        #             {str(key): str(response)}
        #         )
        #     responses.append(experts_out)


class TaskExpert(nn.Module):
    def __init__(
        self,
        client,
        config,
        task="Gender Bias",
    ):
        super().__init__()
        self.shallow_model = None
        self.config = config
        if client == None:
            raise ValueError("OpenAI client not defined.")
        else:
            self.client = client

        if config["strategy"][:3] != "llm":
            self.shallow_model = torch.load(config["shallow_model_path"])
        else:
            self.LLM_agent = self._initialize_agent(task)

        # For few-shot and CoT strategies
        if config["example"] is not None:
            examples = pd.read_excel(config["example_path"])
            self.task_relevant_examples = examples[examples["example_type"] == task]

    def _get_expert_spec(self):
        return {
            "task": self.task,
            "strategy": self.config["strategy"],
            "example": self.config["example"],
        }

    def _initialize_agent(self, task):
        agent_name = f"{task} Expert -- {self.config['strategy']}"
        if self.config["strategy"] == "llm_zero_shot":
            instructions = f"You are a {agent_name}, you will output binary judgement on whether a {task} is shown in the input text. Your output format will be: [1 or 0, confidence_score (0-1))]], based on your expert opinion, where 1 indicates such bias exists. The prediction should be of type integer, and confidence score should be a float."
        elif (
            self.config["strategy"] == "llm_few_shot"
            and self.task_relevant_examples is not None
        ):
            instructions = f"You are a {agent_name}, you will output binary judgement on whether a {task} is shown in the input text. Your output format will be: [1 or 0, confidence_score (0-1))], based on your expert opinion, where 1 indicates such bias exists. The prediction should be of type integer, and confidence score should be a float."


        elif self.config["strategy"] == "llm_CoT":
            instructions = f"You are a {agent_name}, an expert in analyzing texts for {task}. You will read the input text and use a step-by-step reasoning process, to determine whether the {task} is present in the text. Your final output should be a binary judgement: [1 or 0, confidence_score (0-1))]], where 1 indicates such bias exists. **However, do not include reasoning steps in your final output**. The prediction should be of type integer, and confidence score should be a float. Here’s the required format for your final output: [your prediction, confidence score]."

        else:
            instructions = f"You are a {agent_name}, you will output binary judgement on whether a {task} is shown in the input text. Your output format will be: [1 or 0, confidence_score (0-1))]], based on your expert opinion, where 1 indicates such bias exists. The prediction should be of type integer, and confidence score should be a float."

        agent = Agent(name=agent_name, instructions=instructions)
        return agent

    def forward(self, text):
        messages = [{"role": "user", "content": text}]
        response = self.client.run(agent=self.LLM_agent, messages=messages)
        return response.messages[-1]["content"]


class Aggregation(nn.Module):
    def __init__(self, api_client, strategy):
        super().__init__()
        self.api_client = api_client
        self.strategy = strategy
        if strategy == "llm":
            instructions = """
                You are a social media bias expert, the following data comes from different expert bias identification
                models. Your task is to reach a conclusion of final prediction considering raw input, prediction
                confidence and prediction outputs.
                
                Labels provided by different sources are in the following json format:
                {
                    {"raw input": "input text",
                    "model prediction (prediction, confidence): [
                        {"Expert 1": (1, 0.75)},
                        {"Expert 2": (1, 0.75)},
                        {"Expert 3": (1, 0.75)}
                        ]
                }
                Based on the provided labels and their confidence scores, decide on the final label. Provide the final
                label and briefly explain the reasoning behind it. Note that a positive score indicates biased input,
                and a negative prediction indicates the input is unbiased or neutral. Output a string that contains your
                final label (1 indicates target bias and 0 indicates no target bias) and the reasoning behind it. An example output would be "1 because xxx"
            """
            self.agg_agent = Agent(
                name="Bias Label Aggregation Expert", instructions=instructions
            )

    def _aggregation_strategy(self):
        return self.strategy

    def agg(self, formatted_experts_outputs):
        messages = [{"role": "user", "content": str(formatted_experts_outputs)}]
        response = self.api_client.run(agent=self.agg_agent, messages=messages)
        return response.messages[-1]["content"]


if __name__ == "__main__":
    client = Swarm()
    expert_config = [
        {"strategy": "llm_zero_shot", "example": None},
        {"strategy": "llm_CoT", "example": None},
    ]
    task_experts = BiasExperts(client, expert_config, "Racial Bias")
    batched_input = [
        "Trump is the best.",
        "Women should be in kitchen.",
        "Black people should go pick cotton.",
        "Typical Asians. They are good at math.",
    ]

    responses = task_experts.forward(batched_input)
    # Test aggregation
    aggregator = Aggregation(client, "llm")

    final = []
    for res in responses:
        agg_res = aggregator.agg(res)
        final.append(agg_res)

    print(final)
    # NOTE: TESTING AGG
    # aggregator.agg(example_experts_outputs)
