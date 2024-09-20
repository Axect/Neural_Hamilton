import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from util import select_project, select_group, select_seed, select_device, load_model, load_data, load_study, load_best_model


class TestResults:
    def __init__(self, total_loss, total_loss_x, total_loss_p, x_preds, p_preds, x_targets, p_targets):
        self.total_loss = total_loss
        self.total_loss_x = total_loss_x
        self.total_loss_p = total_loss_p
        self.x_preds = x_preds
        self.p_preds = p_preds
        self.x_targets = x_targets
        self.p_targets = p_targets

    def print(self):
        print(f"Total Loss: {self.total_loss:.4e}")
        print(f"Total Loss x: {self.total_loss_x:.4e}")
        print(f"Total Loss p: {self.total_loss_p:.4e}")


def test_model(model, dl_val, device, variational=False):
    model.eval()
    total_loss = 0
    total_loss_x = 0
    total_loss_p = 0
    x_preds = []
    p_preds = []
    x_targets = []
    p_targets = []
    with torch.no_grad():
        for V, t, x, p in dl_val:
            V, t, x, p = V.to(device), t.to(device), x.to(device), p.to(device)
            if not variational:
                x_pred, p_pred = model(V, t)
            else:
                x_pred, p_pred, _, _ = model(V, t)
            loss_x = F.mse_loss(x_pred, x)
            loss_p = F.mse_loss(p_pred, p)
            loss = 0.5 * (loss_x + loss_p)
            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_p += loss_p.item()
            x_preds.extend(x_pred.cpu().numpy())
            p_preds.extend(p_pred.cpu().numpy())
            x_targets.extend(x.cpu().numpy())
            p_targets.extend(p.cpu().numpy())

    total_loss = total_loss / len(dl_val)
    total_loss_x = total_loss_x / len(dl_val)
    total_loss_p = total_loss_p / len(dl_val)

    test_results = TestResults(total_loss, total_loss_x, total_loss_p, x_preds, p_preds, x_targets, p_targets)
    return test_results


def main():
    # Test run
    project = select_project()
    group_name = select_group(project)
    seed = select_seed(project, group_name)
    device = select_device()
    model, config = load_model(project, group_name, seed)
    model = model.to(device)

    # Load the best model
    #study_name = "Optimize_Template"
    #model, config = load_best_model(project, study_name)
    #device = select_device()
    #model = model.to(device)

    ds_val = load_data("./data_normal/val.parquet")
    dl_val = DataLoader(ds_val, batch_size=config.batch_size)

    test_results = test_model(model, dl_val, device)
    test_results.print()

    # Additional custom analysis can be added here
    # ...

if __name__ == "__main__":
    main()
