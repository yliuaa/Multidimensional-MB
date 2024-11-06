import pandas as pd
import numpy as np
import pandas as pd
from swarm import Swarm, Agent
from bias_experts import TaskExpert, Aggregation
from sklearn.metrics import classification_report

# Load the evaluation dataset
eval_file_path = "/home/yifan/Desktop/tidy/PapersNProjects/Code/MMediaBias/Multidimensional-MB/data/reddit_data/eval_dataset(Nov5).csv"
test_dataset = pd.read_csv(eval_file_path)
print(test_dataset)

# Specs
bias_dimensions = [
    "hate speech",
    "linguistic bias",
    "text-level context bias",
    "political bias",
    "racial bias",
]


def load_model():
    client = Swarm()
    prompt_config = {"prompt_setting": ["zero-shot"]}
    model_dict = {
        bias: TaskExpert(client, None, None, prompt_config, bias)
        for bias in bias_dimensions
    }

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

    pass


def perform_inference(model, data):

    for i in range(len(data)):
        predictions = model.forward(data[i])

    return predictions


if __name__ == "__main__":
    # Load the model
    model = load_model()

    # Perform inference on the evaluation data
    predictions = perform_inference(model, test_dataset)

    for bias in bias_dimensions:

        test_dataset[bias] = test_dataset[bias].astype(int)
        true_labels = np.asarray(test_dataset[bias])

        report = classification_report(
            true_labels, predictions, target_names=bias_dimensions
        )

    # Print the evaluation report
    print(report)
