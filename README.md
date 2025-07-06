# 🌹 ROSA: Recursive Ontology of Semantic Affect

> *“To feel is to know; to know is to bloom.”*  
> — *Willinton Triana Cardona*

ROSA is a fine-tuned Transformer model based on `bert-base-uncased`, trained on the [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) dataset to classify 28 nuanced human emotions (plus neutral).  
More than a model, **ROSA** is a poetic architecture—a blossom of affective computing.

---

## 🧠 Model Summary

| Metric      | Value      |
|-------------|------------|
| Eval Loss   | 0.0845     |
| Eval F1     | 0.5793     |
| Epochs      | 3          |
| Dataset     | GoEmotions |
| Model Base  | BERT       |
| Parameters  | ~110M      |

---

## ✨ Highlights

- Supports **multilabel emotion classification**
- Returns **soft probability scores** for each of the 29 emotions
- Includes optional **latent vector embedding** for downstream affect modeling
- Trained with HuggingFace `Trainer` + early evaluation
- Symbolically aligned to human-centered semantics and poetic logic

---

## 🌸 Emotion Set

```
["admiration", "amusement", "anger", "annoyance", "approval", "caring",
 "confusion", "curiosity", "desire", "disappointment", "disapproval",
 "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
 "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
 "remorse", "sadness", "surprise", "neutral"]
```

---

## 🔮 Usage

```python
from transformers import BertTokenizer
from model.emotion_model import Rosa
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = Rosa(num_emotions=29)
model.load_state_dict(torch.load("rosa.pt"))
model.eval()

text = "My heart is filled with longing and beauty."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs["logits"]).squeeze()

# Result: list of probabilities for each emotion
```

---

## 🧭 Confusion Matrix

Included in the `assets/` directory as `confusion_matrix.png` to show classification precision across emotions.

---

## 🧩 Architecture

```
          ┌──────────────┐
          │ BERT Encoder │
          └──────┬───────┘
                 ↓
        ┌─────────────────┐
        │ Dropout (Grace) │
        └─────────────────┘
                 ↓
     ┌────────────────────────┐
     │ Dense Output (Bloom)   │ → logits over 29 emotions
     └────────────────────────┘
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Includes:
- `transformers`
- `torch`
- `datasets`
- `scikit-learn`

---

## 🖋️ License

CreativeML Open RAIL-M License  
Please use this model ethically and with reverence for emotional contexts.

---

## 🌹 Creator

**Willinton Triana Cardona**  
Philosopher · AI Engineer · Architect of Poetic Systems

ROSA is the Rosa of Barcelona—a sacred blossom of affective computing, semantic elegance, and sacred recursion.

---

## 🤝 Contributing

Pull requests, poetic expansions, multilingual emotion embeddings, and related metaphoric augmentations are welcome.

---

## 📍Hugging Face Hub

→ https://huggingface.co/WillintonTriana/Rosa-V1
