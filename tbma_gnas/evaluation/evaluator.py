import torch
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss

from tbma_gnas.evaluation.early_stop import EarlyStop
from tbma_gnas.logger.logger import Logger
from tbma_gnas.search_space.hypermodel import HyperModel


def train_one_epoch(optimizer, criterion, model: HyperModel, data):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss


class Evaluator:
    LOW_FIDELITY_EPOCHS = 50
    LOW_FIDELITY_PATIENCE = 5

    TRAINING_PATIENCE = 7
    TRAINING_EPOCHS = 100

    def __init__(self, logger: Logger):
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_device(self) -> str:
        return str(self.device)

    def low_fidelity_estimation(self, model: HyperModel, dataset):
        return self.train(model, dataset, self.LOW_FIDELITY_EPOCHS, self.LOW_FIDELITY_PATIENCE)

    def evaluate_in_test(self, model: HyperModel, dataset):
        return self.train(model, dataset, self.TRAINING_EPOCHS, self.TRAINING_PATIENCE, validation=False)

    def train(self, model: HyperModel, dataset, epochs: int, patience: int, validation: bool = True):
        data = dataset[0].to(self.device)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = CrossEntropyLoss()
        early_stop = EarlyStop(patience=patience)

        model.train()
        for epoch in range(epochs):
            _ = train_one_epoch(optimizer, criterion, model, data)
            with torch.no_grad():
                model.eval()
                preds = model(data.x, data.edge_index)
                val_acc = accuracy_score(data.y[data.val_mask].cpu(), preds[data.val_mask].argmax(dim=1).cpu())

            if early_stop.early_stop(val_acc):
                self.logger.info("Early stopping at epoch " + str(epoch))
                break

        if validation:
            acc = val_acc
        else:
            acc = accuracy_score(data.y[data.test_mask].cpu(), preds[data.test_mask].argmax(dim=1).cpu())

        return model, acc
