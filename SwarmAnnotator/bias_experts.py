import torch
import torch.nn as nn
import pandas as pd
from swarm import Swarm, Agent


class TaskExpert(nn.Module):
    def __init__(
        self,
        client,
        shallow_model_config,
        example_config,
        prompt_config,
        task="Gender Bias",
    ):
        super().__init__()

        # CONFIGS
        self.prompt_config = prompt_config
        if client == None:
            raise ValueError("OpenAI client not defined.")
        if shallow_model_config is not None:
            self.shallow_model = torch.load(shallow_model_config)
        if example_config is not None:
            examples = pd.read_excel(example_config)
            self.task_relevant_examples = examples[examples["example_type"] == task]

        # NOTE: Use multiple prompting schemes too as experts?
        self.LLM_agent = self.initialize_agent(task)

    def initialize_agent(self, task, specs=None):
        agent_name = f"{task} Expert"
        if self.prompt_config["prompt_setting"] == "zero_shot":
            instructions = f"You are a {agent_name}, you will output binary judgement on whether a {task} is shown in the input text. Your output format will be: [1 or 0], based on your expert opinion, where 1 indicates such bias exists."
        elif (
            self.prompt_config["prompt_setting"] == "few_shot"
            and self.task_relevant_examples is not None
        ):
            instructions = f"You are a {agent_name}, you will output binary judgement on whether a {task} is shown in the input text. Your output format will be: [1 or 0], based on your expert opinion, where 1 indicates such bias exists."
        else:
            instructions = f"You are a {agent_name}, you will output binary judgement on whether a {task} is shown in the input text. Your output format will be: [1 or 0], based on your expert opinion, where 1 indicates such bias exists."

        agent = Agent(name=agent_name, instructions=instructions)
        return agent

    def forward(self, text):
        messages = [{"role": "user", "content": text}]
        response = client.run(agent=self.LLM_agent, messages=messages)
        print(response.messages[-1]["content"])
        return


class Aggregation(nn.Module):
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
        if strategy == "llm":
            instructions = """
                You are a social media bias expert, the following data comes from different expert bias identification
                models. Your task is to reach a conclusion of final prediction considering raw input, prediction
                confidence and prediction outputs.
                
                Labels provided by different sources are in the following json format:
                {
                    {"raw input": "input text"},
                    [{"Expert 1": (1, 0.75)},
                    {"Expert 2": (1, 0.75)},
                    {"Expert 3": (1, 0.75)}]
                }

                Based on the provided labels and their confidence scores, decide on the final label. Provide the final
                label and briefly explain the reasoning behind it. Note that positive score indicates biased input, and
                negative prediction indicates the input is unbiased or neutral.

                Output format:
                Final label: [Positive (1)/Negative (0)]
                Reasoning: 
            """
            self.agg_agent = Agent(
                name="Bias Label Aggregation Expert", instructions=instructions
            )

    def _aggregation_strategy(self):
        return self.strategy

    def agg(self, formatted_experts_outputs):
        messages = [{"role": "user", "content": str(formatted_experts_outputs)}]
        response = client.run(agent=self.agg_agent, messages=messages)
        print(response.messages[-1]["content"])
        return


if __name__ == "__main__":
    client = Swarm()
    prompt_config = {"prompt_setting": "zero-shot"}
    gender_bias_expert = TaskExpert(client, None, None, prompt_config, "Racial Bias")

    # Racial Bias example
    gender_bias_expert.forward("Trump is the best.")
    gender_bias_expert.forward("Women should be in kitchen.")
    gender_bias_expert.forward("Black people should go pick cotton.")
    gender_bias_expert.forward("Typical Asians. They are good at math.")

    # Test aggregation
    aggregator = Aggregation("llm")
    example_experts_outputs = {
        "raw input": "Asians make the best doctors. It's their math skills.",
        "expert predictions (prediction, confidence)": [
            {"RoBERTa-Twitter": (0, 0.9)},
            {"few-shot GPT4o": (0, 0.6)},
            {"zero-shot GPT4o": (1, 0.8)},
            {"CoT GPT4o": (1, 0.3)},
        ],
    }

    aggregator.agg(example_experts_outputs)
