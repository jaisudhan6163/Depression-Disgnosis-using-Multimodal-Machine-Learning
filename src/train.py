"""Training entry point (Tier 1.1, 1.4): proper Dataset/DataLoader, dev-only
model selection, test touched exactly once, 5 seeds minimum with mean +/-
std reporting, plus the four Tier 1.3 baselines.

Usage: python -m src.train [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .data.dataset import DDMMLDataset, build_datasets
from .evaluate import best_threshold, compute_metrics, format_metrics, summarize_seeds
from .models.baselines import joined_turns, majority_class_baseline, tfidf_logreg_baseline
from .models.fusion import MultimodalLSTM, MultimodalTransformerText, TransformerTextOnly, UnimodalLSTM
from .utils import git_sha, set_seed

MODALITIES = ["text", "audio", "video"]
COLLATE_KEYS = ["text", "audio", "video", "audio_mask", "video_mask", "lexical", "label",
                "text_turns_emb", "text_turns_mask"]


def collate(batch: list[dict]) -> dict:
    out = {}
    for key in COLLATE_KEYS:
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    out["participant_id"] = [b["participant_id"] for b in batch]
    return out


def make_loader(ds: DDMMLDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def pos_weight_for(ds: DDMMLDataset) -> torch.Tensor:
    labels = np.array([item["label"].item() for item in ds.items])
    n_pos = max(labels.sum(), 1)
    n_neg = max(len(labels) - labels.sum(), 1)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def forward_fusion(model: MultimodalLSTM, batch: dict) -> torch.Tensor:
    return model(batch["text"], batch["audio"], batch["video"]).squeeze(-1)


def forward_unimodal(model: UnimodalLSTM, batch: dict, modality: str) -> torch.Tensor:
    return model(batch[modality]).squeeze(-1)


def forward_transformer_fusion(model: MultimodalTransformerText, batch: dict) -> torch.Tensor:
    return model(batch["text_turns_emb"], batch["text_turns_mask"], batch["lexical"],
                 batch["audio"], batch["video"]).squeeze(-1)


def forward_transformer_text_only(model: TransformerTextOnly, batch: dict) -> torch.Tensor:
    return model(batch["text_turns_emb"], batch["text_turns_mask"], batch["lexical"]).squeeze(-1)


def train_one_model(model: nn.Module, forward_fn, datasets: dict, config: dict, seed: int) -> dict:
    set_seed(seed)
    train_loader = make_loader(datasets["train"], config["train"]["batch_size"], shuffle=True)
    dev_loader = make_loader(datasets["dev"], config["train"]["batch_size"], shuffle=False)
    test_loader = make_loader(datasets["test"], config["train"]["batch_size"], shuffle=False)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(datasets["train"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"],
                                  weight_decay=config["model"]["weight_decay"])

    best_dev_f1 = -1.0
    best_state = None
    patience = config["train"]["early_stopping_patience"]
    bad_epochs = 0

    for epoch in range(config["train"]["epochs"]):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = forward_fn(model, batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_norm"])
            optimizer.step()

        dev_prob, dev_labels = _predict(model, forward_fn, dev_loader)
        dev_thr = best_threshold(dev_labels, dev_prob)
        dev_metrics = compute_metrics(dev_labels, dev_prob, threshold=dev_thr)

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            marker = "*"
        else:
            bad_epochs += 1
            marker = ""
        print(f"    epoch {epoch+1}: loss={loss.item():.4f} dev_f1={dev_metrics['f1']:.3f}{marker}", flush=True)
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    dev_prob, dev_labels = _predict(model, forward_fn, dev_loader)
    threshold = best_threshold(dev_labels, dev_prob)
    dev_metrics = compute_metrics(dev_labels, dev_prob, threshold=threshold)

    test_prob, test_labels = _predict(model, forward_fn, test_loader)
    test_metrics = compute_metrics(test_labels, test_prob, threshold=threshold)

    return {"dev": dev_metrics, "test": test_metrics, "state_dict": best_state}


@torch.no_grad()
def _predict(model: nn.Module, forward_fn, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for batch in loader:
        logits = forward_fn(model, batch)
        probs.append(torch.sigmoid(logits).numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def run_fusion(datasets: dict, config: dict, seeds: list[int]) -> list[dict]:
    text_dim, audio_dim, video_dim = (
        datasets["train"][0]["text"].shape[-1],
        datasets["train"][0]["audio"].shape[-1],
        datasets["train"][0]["video"].shape[-1],
    )
    results = []
    for seed in seeds:
        set_seed(seed)  # before model construction: seed must also cover weight initialization
        model = MultimodalLSTM(text_dim, audio_dim, video_dim, config["model"]["hidden_size"],
                                config["model"]["output_size"], config["model"]["dropout"])
        res = train_one_model(model, forward_fusion, datasets, config, seed)
        print(f"  seed {seed}: " + format_metrics("fusion/dev", res["dev"]))
        print(f"  seed {seed}: " + format_metrics("fusion/test", res["test"]))
        results.append(res)
    ckpt_dir = Path(config["checkpoints"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results[0]["state_dict"], ckpt_dir / "multimodal_lstm.pth")
    return results


def run_unimodal(modality: str, datasets: dict, config: dict, seeds: list[int]) -> list[dict]:
    dim = datasets["train"][0][modality].shape[-1]
    results = []
    for seed in seeds:
        set_seed(seed)  # before model construction: seed must also cover weight initialization
        model = UnimodalLSTM(dim, config["model"]["hidden_size"], config["model"]["output_size"], config["model"]["dropout"])
        res = train_one_model(model, lambda m, b: forward_unimodal(m, b, modality), datasets, config, seed)
        results.append(res)
    return results


def run_transformer_fusion(datasets: dict, config: dict, seeds: list[int]) -> list[dict]:
    text_hidden = datasets["train"][0]["text_turns_emb"].shape[-1]
    audio_dim = datasets["train"][0]["audio"].shape[-1]
    video_dim = datasets["train"][0]["video"].shape[-1]
    results = []
    for seed in seeds:
        set_seed(seed)
        model = MultimodalTransformerText(text_hidden, audio_dim, video_dim, config["model"]["hidden_size"],
                                           config["model"]["output_size"], config["model"]["dropout"])
        res = train_one_model(model, forward_transformer_fusion, datasets, config, seed)
        print(f"  seed {seed}: " + format_metrics("fusion_transformer/dev", res["dev"]))
        print(f"  seed {seed}: " + format_metrics("fusion_transformer/test", res["test"]))
        results.append(res)
    ckpt_dir = Path(config["checkpoints"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results[0]["state_dict"], ckpt_dir / "multimodal_transformer_text.pth")
    return results


def run_transformer_text_only(datasets: dict, config: dict, seeds: list[int]) -> list[dict]:
    text_hidden = datasets["train"][0]["text_turns_emb"].shape[-1]
    results = []
    for seed in seeds:
        set_seed(seed)
        model = TransformerTextOnly(text_hidden, config["model"]["output_size"], config["model"]["dropout"])
        res = train_one_model(model, forward_transformer_text_only, datasets, config, seed)
        results.append(res)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--transformer-text", action="store_true",
                         help="Also run the Tier 2.1 transformer+attention-pooling text encoder "
                              "(fusion and text-only variants) alongside the Tier 1 word-level-LSTM models.")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    print(f"[train] git SHA: {git_sha()}")

    datasets, _, _ = build_datasets(config, include_transformer_text=args.transformer_text)
    seeds = config["train"]["seeds"]

    print("\n=== Majority class baseline ===")
    train_labels = [item["label"].item() for item in datasets["train"].items]
    dev_labels = [item["label"].item() for item in datasets["dev"].items]
    test_labels = [item["label"].item() for item in datasets["test"].items]
    maj_dev = majority_class_baseline(train_labels, dev_labels)
    maj_test = majority_class_baseline(train_labels, test_labels)
    print(format_metrics("majority/dev", maj_dev))
    print(format_metrics("majority/test", maj_test))

    print("\n=== TF-IDF + LogisticRegression baseline ===")
    train_texts = [joined_turns(item["turns"]) for item in datasets["train"].items]
    dev_texts = [joined_turns(item["turns"]) for item in datasets["dev"].items]
    test_texts = [joined_turns(item["turns"]) for item in datasets["test"].items]
    tfidf_result = tfidf_logreg_baseline(train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels, seed=seeds[0])
    print(format_metrics("tfidf_lr/dev", tfidf_result["dev"]))
    print(format_metrics("tfidf_lr/test", tfidf_result["test"]))

    results_summary = {"majority": {"dev": maj_dev, "test": maj_test}, "tfidf_logreg": tfidf_result}

    print(f"\n=== fusion (multimodal LSTM), {len(seeds)} seeds ===")
    fusion_runs = run_fusion(datasets, config, seeds)
    results_summary["fusion"] = {
        "dev_summary": summarize_seeds([r["dev"] for r in fusion_runs]),
        "test_summary": summarize_seeds([r["test"] for r in fusion_runs]),
        "per_seed": [{"dev": r["dev"], "test": r["test"]} for r in fusion_runs],
    }

    for modality in MODALITIES:
        print(f"\n=== {modality}-only LSTM, {len(seeds)} seeds ===")
        runs = run_unimodal(modality, datasets, config, seeds)
        results_summary[f"{modality}_only"] = {
            "dev_summary": summarize_seeds([r["dev"] for r in runs]),
            "test_summary": summarize_seeds([r["test"] for r in runs]),
            "per_seed": [{"dev": r["dev"], "test": r["test"]} for r in runs],
        }

    if args.transformer_text:
        print(f"\n=== fusion_transformer (transformer text + audio/video LSTM), {len(seeds)} seeds ===")
        ft_runs = run_transformer_fusion(datasets, config, seeds)
        results_summary["fusion_transformer_text"] = {
            "dev_summary": summarize_seeds([r["dev"] for r in ft_runs]),
            "test_summary": summarize_seeds([r["test"] for r in ft_runs]),
            "per_seed": [{"dev": r["dev"], "test": r["test"]} for r in ft_runs],
        }

        print(f"\n=== text_only_transformer, {len(seeds)} seeds ===")
        tt_runs = run_transformer_text_only(datasets, config, seeds)
        results_summary["text_only_transformer"] = {
            "dev_summary": summarize_seeds([r["dev"] for r in tt_runs]),
            "test_summary": summarize_seeds([r["test"] for r in tt_runs]),
            "per_seed": [{"dev": r["dev"], "test": r["test"]} for r in tt_runs],
        }

    Path("results").mkdir(exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\nSaved results/metrics.json")


if __name__ == "__main__":
    main()
