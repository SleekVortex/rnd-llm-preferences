"""Проверки устойчивости ДЗ3 для Qwen3-4B с LoRA.

Обучает large_lora с конфигурацией ДЗ2 (lr 0,00015, две эпохи, все 5352 пары) на заданных seed,
без нового подбора. Для каждого seed сохраняет прогнозы на 1229 парах в исходном порядке ответов
и после перестановки A и B, плюс число токенов в запросе и ответах. Нужна GPU с CUDA.

    python -m experiments.robustness --seeds 42 43 44
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

REPO, REVISION = "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"
LR, EPOCHS = 1.5e-4, 2
CLASSES = ["A", "B", "tie"]
LIMITS = {"prompt": 96, "response_a": 192, "response_b": 192}
BUNDLE = Path("artifacts/hw2-input")
OUT = Path("experiments/reports/hw3/robustness")


def load_data():
    """Повторяет выборки ДЗ2: 5352 обучающие пары и 1229 отложенных групп."""
    train_all = pd.read_json(BUNDLE / "train.jsonl", lines=True, dtype={"id": str}).set_index("id")
    valid_all = pd.read_json(BUNDLE / "valid.jsonl", lines=True, dtype={"id": str}).set_index("id")
    groups = pd.read_csv("experiments/reports/hw2/encoder_pairjudge_tuning/train_groups.csv",
                         dtype={"id": str}).set_index("id").loc[train_all.index]
    train = train_all.loc[~groups.pair_known]
    audit = pd.read_csv(BUNDLE / "split_audit.csv", dtype={"id": str}).set_index("id")
    study = audit.loc[audit.text & ~audit.connected_to_train & ~audit.published_component,
                      ["component", "judge_panel"]].copy()
    study["order"] = [hashlib.sha256(("42|" + i).encode()).hexdigest() for i in study.index]
    study = study.sort_values("order").drop_duplicates("component").drop(columns="order")
    assert len(train) == 5352 and len(study) == 1229 and (~study.judge_panel).sum() == 717
    return train, valid_all.loc[study.index], groups.loc[train.index]


def encode(frame, tokenizer, swap=False):
    """Тот же формат входа, что в ДЗ2. При swap ответы меняются местами."""
    texts = frame[["prompt", "response_a", "response_b"]].map("\n".join)
    if swap:
        texts = texts.rename(columns={"response_a": "response_b", "response_b": "response_a"})[list(LIMITS)]
    headers = [tokenizer.encode(t, add_special_tokens=False) for t in
               ("Question:\n", "\nAnswer A:\n", "\nAnswer B:\n", "\nPreference:")]
    parts = [tokenizer(texts[c].tolist(), add_special_tokens=False, truncation=True, max_length=n).input_ids
             for c, n in LIMITS.items()]
    rows = [headers[0] + q + headers[1] + a + headers[2] + b + headers[3] for q, a, b in zip(*parts)]
    assert max(map(len, rows)) <= 512 and all(s[-1] != tokenizer.pad_token_id for s in rows)
    return rows


def token_counts(frame, tokenizer):
    texts = frame[list(LIMITS)].map("\n".join)
    return pd.DataFrame({c + "_tokens": [len(x) for x in tokenizer(texts[c].tolist(), add_special_tokens=False).input_ids]
                         for c in LIMITS}, index=frame.index)


def new_classifier(pad_id, seed):
    set_seed(seed)
    net = AutoModelForSequenceClassification.from_pretrained(
        REPO, revision=REVISION, num_labels=3, dtype=torch.bfloat16, attn_implementation="sdpa",
        id2label=dict(enumerate(CLASSES)), label2id={v: i for i, v in enumerate(CLASSES)},
        use_safetensors=True, pad_token_id=pad_id)
    net.config.use_cache = False
    initial = torch.empty(net.score.weight.shape, dtype=torch.float32)
    initial.normal_(std=net.config.initializer_range, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        net.score.weight.copy_(initial.to(torch.bfloat16))
    settings = LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.,
                          target_modules=["q_proj", "v_proj"], modules_to_save=["score"])
    net = get_peft_model(net, settings)
    for param in net.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return net.cuda()


def inputs(rows, ids, pad):
    seq = [rows[i] for i in ids]
    width = (max(map(len, seq)) + 7) // 8 * 8
    tokens = np.full((len(seq), width), pad, dtype=np.int64)
    mask = np.zeros_like(tokens)
    for i, s in enumerate(seq):
        tokens[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return {"input_ids": torch.from_numpy(tokens).cuda(), "attention_mask": torch.from_numpy(mask).cuda()}


@torch.inference_mode()
def predict(net, rows, pad):
    net.eval()
    ids = np.arange(len(rows))
    order = sorted(range(len(ids)), key=lambda i: len(rows[i]))
    p = np.empty((len(ids), 3))
    for start in range(0, len(ids), 16):
        loc = order[start:start + 16]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(**inputs(rows, ids[loc], pad)).logits
        p[loc] = logits.float().softmax(-1).cpu().numpy()
    p /= p.sum(1, keepdims=True)
    assert np.isfinite(p).all() and (p >= 0).all()
    return p


def one_epoch(net, opt, rows, y, pad, epoch, seed):
    net.train()
    shuffled = np.random.default_rng(seed + epoch).permutation(len(rows))
    for offset in range(0, len(rows), 64):
        block = sorted(shuffled[offset:offset + 64], key=lambda i: len(rows[i]))
        opt.zero_grad(set_to_none=True)
        for j in range(0, len(block), 8):
            idx = block[j:j + 8]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(**inputs(rows, idx, pad)).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.float(), torch.as_tensor(y[idx], device="cuda"), reduction="sum") / len(block)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.)
        opt.step()


def run(seed, train, test, tok, rows, swapped):
    t = time.perf_counter()
    y = train.target.to_numpy(dtype=np.int64)
    net = new_classifier(tok.pad_token_id, seed)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=LR, weight_decay=.01, fused=True)
    for epoch in range(1, EPOCHS + 1):
        one_epoch(net, opt, rows, y, tok.pad_token_id, epoch, seed)
    original = predict(net, encode(test, tok), tok.pad_token_id)
    # Модель видела ответы в обратном порядке: её p(A) относится к исходному ответу B.
    back = predict(net, swapped, tok.pad_token_id)[:, [1, 0, 2]]
    pd.DataFrame(original, index=test.index, columns=CLASSES).to_csv(OUT / f"seed{seed}.csv")
    pd.DataFrame(back, index=test.index, columns=CLASSES).to_csv(OUT / f"seed{seed}_swap.csv")
    checkpoint = Path("artifacts/hw3-checkpoints") / f"seed{seed}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    net.save_pretrained(checkpoint, safe_serialization=True)
    cfg = dict(model=REPO, revision=REVISION, mode="large_lora", seed=seed, lr=LR, epochs=EPOCHS,
               train_rows=len(train), test_rows=len(test), torch=str(torch.__version__),
               gpu=torch.cuda.get_device_name(), seconds=round(time.perf_counter() - t))
    (OUT / f"seed{seed}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("seed", seed, "готово за", cfg["seconds"], "с", flush=True)
    del net, opt
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    OUT.mkdir(parents=True, exist_ok=True)
    train, test, _ = load_data()
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    token_counts(test, tok).to_csv(OUT.parent / "tokens.csv")
    rows, swapped = encode(train, tok), encode(test, tok, swap=True)
    for seed in args.seeds:
        run(seed, train, test, tok, rows, swapped)


if __name__ == "__main__":
    main()
