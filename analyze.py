import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from util import select_group, select_seed, select_device, load_model, load_data, load_study, load_best_model

def test_model(model, dl_val, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in dl_val:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y)
            total_loss += loss.item()
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    return total_loss / len(dl_val), all_preds, all_targets

def main():
    # Test run
    project = "PyTorch_Template"
    #group_name = select_group(project)
    #seed = select_seed(project, group_name)
    #device = select_device()
    #model, config = load_model(project, group_name, seed)
    #model = model.to(device)

    # Load the best model
    study_name = "Optimize_Template"
    model, config = load_best_model(project, study_name)
    device = select_device()
    model = model.to(device)

    _, dl_val = load_data()  # Assuming this is implemented in util.py

    val_loss, preds, targets = test_model(model, dl_val, device)
    print(f"Validation Loss: {val_loss}")

    # Additional custom analysis can be added here
    # ...

if __name__ == "__main__":
    main()
