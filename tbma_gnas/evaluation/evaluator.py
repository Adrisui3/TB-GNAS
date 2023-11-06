import torch
import gc
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
    LOW_FIDELITY_EPOCHS = 25
    LOW_FIDELITY_PATIENCE = 5

    TRAINING_PATIENCE = 7
    TRAINING_EPOCHS = 100

    def __init__(self, logger: Logger, dataset):
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = dataset[0].to(self.device)

    def get_device(self) -> str:
        return str(self.device)

    def low_fidelity_estimation(self, model: HyperModel):
        return self.train(model, self.LOW_FIDELITY_EPOCHS, self.LOW_FIDELITY_PATIENCE)

    def evaluate_in_test(self, model: HyperModel):
        return self.train(model, self.TRAINING_EPOCHS, self.TRAINING_PATIENCE, validation=False)

    def train(self, model: HyperModel, epochs: int, patience: int, validation: bool = True):
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = CrossEntropyLoss()
        early_stop = EarlyStop(patience=patience)

        model.train()
        for epoch in range(epochs):
            _ = train_one_epoch(optimizer, criterion, model, self.data)
            with torch.no_grad():
                model.eval()
                preds = model(self.data.x, self.data.edge_index)
                val_acc = accuracy_score(self.data.y[self.data.val_mask].cpu(), preds[self.data.val_mask].argmax(dim=1).cpu())

            if early_stop.early_stop(val_acc):
                self.logger.info("Early stopping at epoch " + str(epoch))
                break

        if validation:
            acc = val_acc
        else:
            acc = accuracy_score(self.data.y[self.data.test_mask].cpu(), preds[self.data.test_mask].argmax(dim=1).cpu())

        torch.cuda.empty_cache()
        gc.collect()

        return model, acc
