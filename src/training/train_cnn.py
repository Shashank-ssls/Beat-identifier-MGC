"""Train the small CNN on mel-spectrograms, with AMP + early stopping + MLflow.

## What this code does
The deep-learning half of the benchmark. It:
1. Builds DataLoaders over the cached spectrograms (same committed split).
2. Trains the SmallCNN with the AdamW optimizer and cross-entropy loss, applying
   light SpecAugment to each training batch.
3. Uses **mixed precision (AMP)** on GPU — does math in float16 where safe, which
   roughly halves memory and speeds training (a deliberate small-VRAM tactic
   kept even though Kaggle has 16GB). It's a no-op on CPU.
4. Tracks validation macro-F1 each epoch and keeps the BEST weights
   (**early stopping** if val stops improving for `patience` epochs).
5. Evaluates the best model on the test set, logs everything to MLflow, and
   saves the weights to `models/cnn.pt` for Phase 5/6.

## Where it runs
- Local CPU **smoke test** (prove it runs):
    python -m src.training.train_cnn --device cpu --epochs 2 --subset 64
- Real training on **Kaggle GPU** (16GB): run with defaults; afterwards download
  `models/cnn.pt` as a Kaggle output and commit it / register in local MLflow.
"""

from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.manifest import label_maps
from src.data.melspec_dataset import MelspecDataset
from src.evaluation.metrics import compute_metrics, save_confusion_matrix
from src.models.cnn import SpecAugment, build_cnn
from src.utils import PROJECT_ROOT, load_config, set_seed

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return (mean_loss, y_true, y_pred) over a loader."""
    model.eval()
    losses, trues, preds = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.cpu().numpy())
    return (
        float(np.mean(losses)),
        np.concatenate(trues),
        np.concatenate(preds),
    )


def train(args) -> None:
    cfg = load_config("cnn")
    tcfg = cfg["training"]
    set_seed(cfg["seed"])

    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    batch_size = args.batch_size or tcfg["batch_size"]
    epochs = args.epochs or tcfg["epochs"]
    print(f"device={device}  batch_size={batch_size}  epochs={epochs}  subset={args.subset}")

    # --- Data ---
    train_ds = MelspecDataset("train", limit=args.subset)
    val_ds = MelspecDataset("val", limit=args.subset)
    test_ds = MelspecDataset("test", limit=args.subset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=args.num_workers)

    # --- Model / optim ---
    model = build_cnn(cfg["model"]).to(device)
    aug_cfg = cfg["augmentation"]
    augment = (
        SpecAugment(time_mask_param=aug_cfg["time_mask_param"],
                    freq_mask_param=aug_cfg["freq_mask_param"]).to(device)
        if aug_cfg["enabled"] else None
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                                  weight_decay=tcfg["weight_decay"])
    use_amp = bool(tcfg["amp"]) and use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    _, idx_to_genre = label_maps()
    class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    best_f1, best_state, patience_left = -1.0, None, tcfg["early_stopping_patience"]

    with mlflow.start_run(run_name="cnn"):
        mlflow.log_params({f"model__{k}": v for k, v in cfg["model"].items()})
        mlflow.log_params({
            "batch_size": batch_size, "epochs": epochs, "lr": tcfg["learning_rate"],
            "weight_decay": tcfg["weight_decay"], "amp": use_amp, "device": device.type,
        })

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                if augment is not None:
                    x = augment(x)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(loss.item())

            val_loss, val_true, val_pred = evaluate(model, val_loader, criterion, device)
            val_m = compute_metrics(val_true, val_pred, class_names)
            mlflow.log_metrics({
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_accuracy": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
            }, step=epoch)
            print(f"  epoch {epoch:3d}  train_loss={np.mean(train_losses):.3f}  "
                  f"val_loss={val_loss:.3f}  val_acc={val_m['accuracy']:.3f}  "
                  f"val_macroF1={val_m['macro_f1']:.3f}")

            # --- early stopping on val macro-F1 ---
            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = tcfg["early_stopping_patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  early stopping at epoch {epoch} (best val_macroF1={best_f1:.3f})")
                    break

        # --- final test eval with best weights ---
        if best_state is not None:
            model.load_state_dict(best_state)
        _, test_true, test_pred = evaluate(model, test_loader, criterion, device)
        test_m = compute_metrics(test_true, test_pred, class_names)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})

        cm_path = save_confusion_matrix(test_true, test_pred, class_names,
                                        REPORTS_DIR / "cm_cnn.png",
                                        title="CNN — test confusion matrix")
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # --- persist weights + arch for Phase 5/6 ---
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / "cnn.pt")
        (MODELS_DIR / "cnn_arch.json").write_text(json.dumps(cfg["model"], indent=2))
        mlflow.log_artifact(str(MODELS_DIR / "cnn.pt"), artifact_path="model")

        print(f"\n  CNN  test_acc={test_m['accuracy']:.3f}  test_macroF1={test_m['macro_f1']:.3f}")
        print(f"  saved weights -> {MODELS_DIR / 'cnn.pt'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the SmallCNN on mel-spectrograms.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--subset", type=int, default=None, help="limit clips per split (smoke test)")
    p.add_argument("--num-workers", type=int, default=0)
    train(p.parse_args())


if __name__ == "__main__":
    main()
