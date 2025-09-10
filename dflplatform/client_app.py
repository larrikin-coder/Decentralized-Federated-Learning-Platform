# # client_app.py
# from flwr.client import ClientApp, NumPyClient
# from dflplatform.task import MRINet, train, test, get_weights, set_weights, load_data
# import torch


from flwr.client import ClientApp, NumPyClient
from dflplatform.task import LinearRegressionModel, train_model, test_model, get_weights, set_weights, load_salary_data
import torch


class FlowerClient(NumPyClient):
    def __init__(self, csv_path):
        self.model = LinearRegressionModel(input_dim=1)
        self.csv_path = csv_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_weights(self.model)
    
    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        trainloader, _ = load_salary_data(self.csv_path)
        instructions = config.get("training_instructions", {"epochs": 5, "lr": 0.01, "optimizer_type": "SGD"})
        loss = train_model(self.model, trainloader, instructions, self.device)
        return get_weights(self.model), len(trainloader.dataset), {"loss": loss}
    
    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        _, testloader = load_salary_data(self.csv_path)
        loss = test_model(self.model, testloader, self.device)
        return loss, len(testloader.dataset), {"mse_loss": loss}


def client_fn(context):
    node_config = context.node_config or {}
    csv_path = node_config.get("csv_path")
    training_cfg = node_config.get("training_instructions")
    return FlowerClient(csv_path)


app = ClientApp(client_fn=client_fn)


