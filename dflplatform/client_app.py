# # client_app.py
# from flwr.client import ClientApp, NumPyClient
# from dflplatform.task import MRINet, train, test, get_weights, set_weights, load_data
# import torch


from flwr.client import ClientApp, NumPyClient
from dflplatform.task import LinearRegressionModel, train_model, test_model, get_weights, set_weights, load_salary_data
import torch

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # <-- override Flower's default logging
)

logger = logging.getLogger("client")
logger.setLevel(logging.INFO)



class FlowerClient(NumPyClient):
    def __init__(self, csv_path):
        self.model = LinearRegressionModel(input_dim=1)
        self.csv_path = csv_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"client initialized with dataset: {csv_path}")
    def get_parameters(self, config):
        return get_weights(self.model)
    
    def fit(self, parameters, config):
        logger.info(f"[FIT] Starting training on {self.csv_path}")
        set_weights(self.model, parameters)
        trainloader, _ = load_salary_data(self.csv_path)
        instructions = config.get("training_instructions", {"epochs": 5, "lr": 0.01, "optimizer_type": "SGD"})
        logger.info(f"[FIT] Training instructions: {instructions}")
        loss = train_model(self.model, trainloader, instructions, self.device)
        logger.info(f"[FIT] Finished training on {self.csv_path}, loss={loss}")
        return get_weights(self.model), len(trainloader.dataset), {"loss": loss}
    
    def evaluate(self, parameters, config):
        logger.info(f"[EVAL] Starting evaluation on {self.csv_path}")
        set_weights(self.model, parameters)
        _, testloader = load_salary_data(self.csv_path)
        loss = test_model(self.model, testloader, self.device)
        logger.info(f"[EVAL] Finished evaluation on {self.csv_path}, mse_loss={loss}")
        return loss, len(testloader.dataset), {"mse_loss": loss}


def client_fn(context):
    node_config = context.node_config or {}
    csv_path = node_config.get("csv_path")
    training_cfg = node_config.get("training_instructions")
    return FlowerClient(csv_path)


app = ClientApp(client_fn=client_fn)


