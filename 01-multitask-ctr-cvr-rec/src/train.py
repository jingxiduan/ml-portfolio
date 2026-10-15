import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CtrCvrDataset
from model import RankMixerMTL
from metrics import gauc, binary_metrics

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    for sparse_x, dense_x, y_click, y_convert in loader:
        sparse_x, dense_x = sparse_x.to(DEVICE), dense_x.to(DEVICE)
        y_click, y_convert = y_click.to(DEVICE), y_convert.to(DEVICE)

        ctr_logit, cvr_logit = model(sparse_x, dense_x)
        ctr_loss = bce(ctr_logit, y_click).mean()

        mask = (y_convert >= 0).float()
        y_convert_safe = torch.clamp(y_convert, min=0)
        cvr_loss_raw = bce(cvr_logit, y_convert_safe)
        cvr_loss = (cvr_loss_raw * mask).sum() / (mask.sum() + 1e-6)

        loss = ctr_loss + cvr_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_user, all_click_y, all_click_p = [], [], []
    all_convert_y, all_convert_p = [], []

    for sparse_x, dense_x, y_click, y_convert in loader:
        uid = sparse_x[:, 0].numpy()
        sparse_x_d, dense_x_d = sparse_x.to(DEVICE), dense_x.to(DEVICE)
        ctr_logit, cvr_logit = model(sparse_x_d, dense_x_d)
        ctr_p = torch.sigmoid(ctr_logit).cpu().numpy()
        cvr_p = torch.sigmoid(cvr_logit).cpu().numpy()

        all_user.append(uid)
        all_click_y.append(y_click.numpy())
        all_click_p.append(ctr_p)

        mask = (y_convert.numpy() >= 0)
        if mask.any():
            all_convert_y.append(y_convert.numpy()[mask])
            all_convert_p.append(cvr_p[mask])

    all_user = np.concatenate(all_user)
    all_click_y = np.concatenate(all_click_y)
    all_click_p = np.concatenate(all_click_p)
    all_convert_y = np.concatenate(all_convert_y)
    all_convert_p = np.concatenate(all_convert_p)

    ctr_metrics = binary_metrics(all_click_y, all_click_p)
    ctr_metrics["gauc"] = gauc(all_user, all_click_y, all_click_p)

    cvr_metrics = binary_metrics(all_convert_y, all_convert_p)

    return {"ctr": ctr_metrics, "cvr": cvr_metrics}


def main(epochs=3, batch_size=2048, lr=1e-3, embed_dim=16, num_mixer_blocks=2):
    train_ds = CtrCvrDataset("train")
    stats = train_ds.stats
    val_ds = CtrCvrDataset("val", stats=stats)
    test_ds = CtrCvrDataset("test", stats=stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)

    model = RankMixerMTL(
        vocab_size=stats["vocab_size"],
        sparse_cols=train_ds.sparse_cols,
        dense_dim=len(train_ds.dense_cols),
        embed_dim=embed_dim,
        num_mixer_blocks=num_mixer_blocks,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer)
        val_metrics = evaluate(model, val_loader)
        dt = time.time() - t0
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_ctr_auc={val_metrics['ctr']['auc']:.4f} val_ctr_gauc={val_metrics['ctr']['gauc']:.4f} "
              f"val_cvr_auc={val_metrics['cvr']['auc']:.4f} time={dt:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})

    test_metrics = evaluate(model, test_loader)
    print("Final test metrics:", json.dumps(test_metrics, indent=2, ensure_ascii=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"history": history, "test": test_metrics}, f, ensure_ascii=False, indent=2)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs_x = [h["epoch"] for h in history]
        ctr_auc = [h["val"]["ctr"]["auc"] for h in history]
        cvr_auc = [h["val"]["cvr"]["auc"] for h in history]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_x, ctr_auc, marker="o", label="val CTR AUC")
        plt.plot(epochs_x, cvr_auc, marker="s", label="val CVR AUC")
        plt.xlabel("epoch")
        plt.ylabel("AUC")
        plt.title("Multi-task CTR/CVR training curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)

    return test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--num_mixer_blocks", type=int, default=2)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.embed_dim, args.num_mixer_blocks)
