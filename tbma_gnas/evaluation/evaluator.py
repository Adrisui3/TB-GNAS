import torch
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss

from tbma_gnas.search_space.hypermodel import HyperModel


class Evaluator:
    LOW_FIDELITY_EPOCHS = 25

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_device(self) -> str:
        return str(self.device)

    def low_fidelity_estimation(self, model: HyperModel, dataset, verbose: bool = False):
        print(model.get_blocks())
        data = dataset[0].to(self.device)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = CrossEntropyLoss()

        model.train()
        for epoch in range(self.LOW_FIDELITY_EPOCHS):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if verbose:
                print("Epoch:", epoch, "--- Loss:", loss.item())

        with torch.no_grad():
            model.eval()
            preds = model(data.x, data.edge_index)
        val_acc = accuracy_score(data.y[data.val_mask].cpu(), preds[data.val_mask].argmax(dim=1).cpu())

        return model, val_acc
