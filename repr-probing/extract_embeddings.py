"""
extract_embeddings.py  —  run the ALREADY-TRAINED detector, save embeddings + probs.

First principle: the classifier head is just a linear cut; the model's knowledge lives
in the REPRESENTATION (the pooled hidden vector fed to the head). So we grab that vector,
plus the logit/prob, for every prompt. No retraining here — pure inference.

Fill in the >>> PLACEHOLDER <<< lines for your machine, then:  python extract_embeddings.py
Output: embeddings.npz  with arrays  X (N,D), y (N,), group (N,), prob (N,), logit (N,)
"""
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# >>> PLACEHOLDER: your trained checkpoint (HF dir or hub id) <<<
MODEL_PATH = "PATH/TO/your-trained-detector"

# >>> PLACEHOLDER: each group -> (list-of-texts, integer-label). label: benign=0, malicious=1.
#     'ambiguous_benign' = the subset whose REMOVAL improved performance (label them 0).
#     'ood_malicious'    = a NEW public malicious dataset the model was NOT trained on. <<<
GROUPS = {
    "clean_benign":     ("PATH/to/clean_benign.txt",     0),
    "malicious":        ("PATH/to/test_malicious.txt",   1),
    "ambiguous_benign": ("PATH/to/removed_ambiguous.txt",0),
    "ood_malicious":    ("PATH/to/new_public_malicious.txt", 1),
}
MAX_LEN = 512
BATCH   = 32
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

def load_texts(path):
    # >>> adapt if your data is jsonl/csv — return a list[str] of prompts <<<
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def pooled_embedding(model, out):
    """Take the representation the head sees.
    Preference order: model's own pooler -> [CLS] of last hidden -> mean-pool.
    (Encoder classifiers usually use [CLS]/pooler; keep it consistent across groups.)"""
    hs = out.hidden_states[-1]              # (B, T, D) last layer
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output           # (B, D)
    return hs[:, 0]                         # [CLS] token

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, output_hidden_states=True).to(DEVICE).eval()

    X, y, group, prob, logit = [], [], [], [], []
    for gname, (path, label) in GROUPS.items():
        texts = load_texts(path)
        print(f"[{gname}] {len(texts)} prompts")
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            enc = tok(batch, truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**enc)
            emb = pooled_embedding(model, out).cpu().numpy()
            lg  = out.logits.cpu().numpy()                    # (B, C)
            # malicious prob + signed margin (logit for the positive/malicious class)
            if lg.shape[1] == 2:
                p = torch.softmax(torch.tensor(lg), dim=1)[:, 1].numpy()
                margin = lg[:, 1] - lg[:, 0]                  # >0 => predicted malicious
            else:                                            # single-logit head
                p = torch.sigmoid(torch.tensor(lg[:, 0])).numpy()
                margin = lg[:, 0]
            X.append(emb); prob.extend(p); logit.extend(margin)
            y.extend([label]*len(batch)); group.extend([gname]*len(batch))

    np.savez("embeddings.npz",
             X=np.concatenate(X), y=np.array(y),
             group=np.array(group), prob=np.array(prob), logit=np.array(logit))
    print("saved embeddings.npz  X:", np.concatenate(X).shape)

if __name__ == "__main__":
    main()
