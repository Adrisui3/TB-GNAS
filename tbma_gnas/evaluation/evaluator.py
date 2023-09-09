import torch
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss

from tbma_gnas.search_space.hypermodel import HyperModel


def train_one_epoch(optimizer, criterion, model: HyperModel, data):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss


class Evaluator:
    LOW_FIDELITY_EPOCHS = 25
    TRAINING_EPOCHS = 100

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_device(self) -> str:
        return str(self.device)

    def low_fidelity_estimation(self, model: HyperModel, dataset, verbose: bool = False):
        return self.train(model, dataset, self.LOW_FIDELITY_EPOCHS, verbose=verbose)

    def evaluate_in_test(self, model: HyperModel, dataset, verbose: bool = False):
        return self.train(model, dataset, self.TRAINING_EPOCHS, validation=False, verbose=verbose)

    # TODO: Improve this estimation by using early stopping and modularize iterations as shown in https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def train(self, model: HyperModel, dataset, epochs: int, validation: bool = True, verbose: bool = False):
        data = dataset[0].to(self.device)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            loss = train_one_epoch(optimizer, criterion, model, data)
            if verbose:
                print("Epoch:", epoch, "--- Loss:", loss.item())

        with torch.no_grad():
            model.eval()
            preds = model(data.x, data.edge_index)

        if validation:
            acc = accuracy_score(data.y[data.val_mask].cpu(), preds[data.val_mask].argmax(dim=1).cpu())
        else:
            acc = accuracy_score(data.y[data.test_mask].cpu(), preds[data.test_mask].argmax(dim=1).cpu())

        return model, acc
