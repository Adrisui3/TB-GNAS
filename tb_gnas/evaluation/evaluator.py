import gc

import torch
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss

from tb_gnas.evaluation.early_stop import EarlyStop
from tb_gnas.logger.logger import Logger
from tb_gnas.search_space.hypermodel import HyperModel


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

    TRAINING_PATIENCE = 10
    TRAINING_EPOCHS = 100

    ADAM_PARAMETERS = {"lr": {
        "PubMed": 0.01,
        "Cora": 0.01,
        "Citeseer": 0.01},
        "weight_decay": {
            "PubMed": 5e-4,
            "Cora": 5e-4,
            "Citeseer": 5e-4
        }
    }

    def __init__(self, logger: Logger, dataset):
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = dataset[0].to(self.device)
        self.lr = self.ADAM_PARAMETERS["lr"][dataset.name]
        self.weight_decay = self.ADAM_PARAMETERS["weight_decay"][dataset.name]

    def get_device(self) -> str:
        return str(self.device)

    def low_fidelity_estimation(self, model: HyperModel):
        return self.train(model, self.LOW_FIDELITY_EPOCHS, self.LOW_FIDELITY_PATIENCE)

    def evaluate_in_test(self, model: HyperModel):
        return self.train(model, self.TRAINING_EPOCHS, self.TRAINING_PATIENCE, validation=False)

    def train(self, model: HyperModel, epochs: int, patience: int, validation: bool = True):
        torch.cuda.empty_cache()
        gc.collect()

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
