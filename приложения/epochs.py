import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, set_seed
from pairjudge import PairwiseJudge
from peft import LoraConfig, get_peft_model


bundle = Path("artifacts/hw2-input")
reports = Path("приложения/reports/hw2-extra")
sources = {
    "mbert": ("google-bert/bert-base-multilingual-cased", "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"),
    "roberta": ("FacebookAI/roberta-large", "722cf37b1afa9454edce342e7895e588b6ff1d59"),
    "modern": ("answerdotai/ModernBERT-large", "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"),
    "pair": ("DaoyuanLi/pairjudge-qwen2.5-0.5b-v0.3", "aa9dfb488147c45572921a3a93d00864dc1f526d"),
}


def encode_rows(frame, tok, judge=None):
    fields = ["prompt", "response_a", "response_b"]
    if judge is not None:
        packed = judge._pack_frame(frame[fields])
        return [{"input_ids": ids, "labels": int(y)} for ids, y in zip(packed.input_ids, frame.target)]
    parts = []
    for column, limit in zip(fields, [96, 206, 206]):
        strings = frame[column].map("\n".join).tolist()
        parts.append(tok(strings, max_length=limit, truncation=True, add_special_tokens=False)["input_ids"])
    return [{"input_ids": [tok.cls_token_id] + q + [tok.sep_token_id] + a + [tok.sep_token_id] + b + [tok.sep_token_id],
             "labels": int(y)} for q, a, b, y in zip(*parts, frame.target)]


def load_model(name):
    set_seed(42)
    source, revision = sources[name]
    if name == "pair":
        judge = PairwiseJudge.from_pretrained(source, revision=revision, device="cuda")
        assert list(judge.label_order) == ["a_wins", "b_wins", "tie"]
        lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                          modules_to_save=["score"], task_type="SEQ_CLS")
        net = get_peft_model(judge.model, lora)
        return net, judge.tokenizer, judge
    tok = AutoTokenizer.from_pretrained(source, revision=revision)
    net = AutoModelForSequenceClassification.from_pretrained(source, revision=revision, num_labels=3,
            use_safetensors=True, attn_implementation="sdpa")
    return net.cuda(), tok, None


def loader(rows, tok, batch, epoch=None):
    return DataLoader(rows, batch_size=batch, shuffle=epoch is not None,
                      generator=torch.Generator().manual_seed(42 + (epoch or 0)),
                      collate_fn=DataCollatorWithPadding(tok, pad_to_multiple_of=8))

@torch.inference_mode()
def predict(net, rows, tok, micro):
    net.eval()
    result = []
    for batch in loader(rows, tok, micro):
        batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(**batch.to("cuda")).logits
        result.extend(logits.float().softmax(-1).cpu().numpy())
    return np.array(result, dtype=float)

def mean_loss(rows, p):
    labels = [r["labels"] for r in rows]
    p = p / p.sum(1, keepdims=True)
    return float(-np.log(np.clip(p[np.arange(len(rows)), labels], np.finfo(float).eps, 1)).mean())


def train_epoch(net, rows, tok, micro, opt, epoch):
    net.train()
    opt.zero_grad(set_to_none=True)
    seen = 0
    for batch in loader(rows, tok, micro, epoch):
        size = len(batch["labels"])
        remaining = len(rows) - seen + (seen % 64)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net(**batch.to("cuda")).loss * (size / min(64, remaining))
        loss.backward()
        seen += size
        if seen % 64 == 0 or seen == len(rows):
            opt.step()
            opt.zero_grad(set_to_none=True)


def optimizer(net, name):
    lr = 1e-4 if name == "pair" else 2e-5
    return torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=lr, weight_decay=.01)

def study_epochs(name, texts_train, meta, fit_ix, dev_ix):
    net, tok, judge = load_model(name)
    rows = encode_rows(texts_train, tok, judge)
    allowed = ~meta.pair_known if name == "pair" else pd.Series(True, index=meta.index)
    fit = [rows[i] for i in fit_ix if allowed.iloc[i]]
    dev = [rows[i] for i in dev_ix if allowed.iloc[i]]
    opt = optimizer(net, name)
    micro = 16 if name == "mbert" else 8
    history = []
    for epoch in range(1, 3 if name == "pair" else 5):
        train_epoch(net, fit, tok, micro, opt, epoch)
        train_loss = mean_loss(fit, predict(net, fit, tok, micro))
        dev_loss = mean_loss(dev, predict(net, dev, tok, micro))
        history.append((epoch, len(fit), len(dev), train_loss, dev_loss))
        print(name, epoch, round(train_loss, 5), round(dev_loss, 5), flush=True)
    history = pd.DataFrame(history, columns=["epoch", "n_train", "n_dev", "train_loss", "dev_loss"])
    history.to_csv(reports / (name + "_curve.csv"), index=False)
    return history


def final_predictions(name, history, texts_train, texts_valid, meta, panel):
    best = int(history.loc[history.dev_loss.idxmin(), "epoch"])
    epochs = {"best": best}
    if name == "mbert":
        later = history.loc[history.epoch >= 2]
        epochs.update(first=1, longer=int(later.loc[later.dev_loss.idxmin(), "epoch"]))
    net, tok, judge = load_model(name)
    train = texts_train.loc[~meta.pair_known] if name == "pair" else texts_train
    valid = texts_valid.loc[panel] if name == "pair" else texts_valid
    fit, test = encode_rows(train, tok, judge), encode_rows(valid, tok, judge)
    micro = 16 if name == "mbert" else 8
    opt = optimizer(net, name)
    def save(tag, p):
        f = pd.DataFrame(p, index=valid.index, columns=["A", "B", "tie"])
        assert f.index.is_unique and np.allclose(f.sum(axis=1), 1, atol=1e-5)
        f.to_csv(reports / (name + "_" + tag + ".csv"))
    if name == "pair":
        judge.model = net
        def native_probs():
            return judge.predict_proba(valid[["prompt", "response_a", "response_b"]], swap_debias=False, batch_size=8)
        save("native_frozen", native_probs())
        save("frozen", predict(net, test, tok, micro))
    for epoch in range(1, max(epochs.values()) + 1):
        train_epoch(net, fit, tok, micro, opt, epoch)
        if epoch in epochs.values():
            p = predict(net, test, tok, micro)
            for tag, picked in epochs.items():
                if picked == epoch:
                    save(tag, p)
    if name == "pair":
        save("native_best", native_probs())
    print(name, "выбор эпох", epochs, flush=True)


def run_models(names):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    reports.mkdir(parents=True, exist_ok=True)
    texts_train = pd.read_json(bundle / "train.jsonl", lines=True, dtype={"id": str}).set_index("id")
    texts_valid = pd.read_json(bundle / "valid.jsonl", lines=True, dtype={"id": str}).set_index("id")
    meta = pd.read_csv(reports / "train_groups.csv", dtype={"id": str}).set_index("id").loc[texts_train.index]
    fold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fit_ix, dev_ix = next(fold.split(meta, texts_train.target, meta.component))
    assert not set(meta.component.iloc[fit_ix]) & set(meta.component.iloc[dev_ix])
    assert meta.index.is_unique and len(meta) == 6795
    panel = pd.read_csv(bundle / "judge_ids.csv", dtype={"id": str}).id
    print("Внутри train", len(fit_ix), "обучение,", len(dev_ix), "проверка")
    for name in names:
        history = study_epochs(name, texts_train, meta, fit_ix, dev_ix)
        gc.collect()
        torch.cuda.empty_cache()
        final_predictions(name, history, texts_train, texts_valid, meta, panel)
        gc.collect()
        torch.cuda.empty_cache()
