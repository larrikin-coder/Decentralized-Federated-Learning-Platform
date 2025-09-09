# client_app.py
from flwr.client import ClientApp, NumPyClient
from dflplatform.task import MRINet, train, test, get_weights, set_weights, load_data
import torch


class FlowerClient(NumPyClient):
    def __init__(self, partition_dir):
        self.model = MRINet()
        self.partition_dir = partition_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        trainloader, _ = load_data(self.partition_dir)
        loss = train(self.model, trainloader, epochs=1, device=self.device)
        return get_weights(self.model), len(trainloader.dataset), {"loss": loss}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        _, testloader = load_data(self.partition_dir)
        loss, accuracy = test(self.model, testloader, self.device)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(context):
    partition_dir = context.node_config["partition_dir"]
    return FlowerClient(partition_dir)


app = ClientApp(client_fn=client_fn)
