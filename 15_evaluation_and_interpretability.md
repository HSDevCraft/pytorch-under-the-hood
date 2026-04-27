# Module 15: Evaluation & Model Interpretability

## Learning Objectives
By the end of this module you will be able to:
- Select and implement the right evaluation metrics for every ML task
- Apply Grad-CAM, Integrated Gradients, and SHAP for visual explanations
- Use attention weight visualization for transformer models
- Audit models for fairness and calibration
- Implement confusion matrix analysis, ROC/PR curves, and confidence calibration
- Use `torchmetrics` for distributed-safe metric computation
- Write model cards and evaluation reports for production models

---

## 15.1 Evaluation Metrics by Task

### Classification

```python
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list = None,
) -> dict:
    """Full classification evaluation suite."""
    model.eval()
    all_logits, all_preds, all_labels = [], [], []

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds  = logits.argmax(-1)
            all_logits.append(logits.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits).numpy()
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

    n_cls  = logits.shape[-1]

    # Basic metrics
    acc = accuracy_score(labels, preds)
    p, r, f1, support = precision_recall_fscore_support(labels, preds, average="macro")

    # Per-class metrics
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)

    # ROC-AUC (one-vs-rest for multiclass)
    if n_cls == 2:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")

    return {
        "accuracy":     acc,
        "precision":    p,
        "recall":       r,
        "f1_macro":     f1,
        "roc_auc":      auc,
        "per_class":    report,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str = "Confusion Matrix"):
    """Normalized confusion matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regressor(preds: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "mse":   mean_squared_error(targets, preds),
        "rmse":  mean_squared_error(targets, preds) ** 0.5,
        "mae":   mean_absolute_error(targets, preds),
        "r2":    r2_score(targets, preds),
        "mape":  np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100,
    }
```

### NLP Metrics

```python
# pip install evaluate rouge_score sacrebleu bert_score
import evaluate

# BLEU score for translation
bleu = evaluate.load("sacrebleu")
predictions = ["The cat is on the mat."]
references  = [["The cat is on the mat."]]
result = bleu.compute(predictions=predictions, references=references)
print(f"BLEU: {result['score']:.2f}")

# ROUGE for summarization
rouge = evaluate.load("rouge")
result = rouge.compute(predictions=predictions, references=references)
print(result)  # rouge1, rouge2, rougeL, rougeLsum

# BERTScore (semantic similarity)
bertscore = evaluate.load("bertscore")
result = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en",
)
print(f"BERTScore F1: {result['f1'][0]:.4f}")

# Perplexity for language models
def compute_perplexity(model, loader, device):
    model.eval()
    total_loss = total_tokens = 0
    with torch.inference_mode():
        for x in loader:
            x = x.to(device)
            logits = model(x[:, :-1])
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_loss   += loss.item()
            total_tokens += (x.size(1) - 1) * x.size(0)
    return np.exp(total_loss / total_tokens)
```

---

## 15.2 torchmetrics for Distributed Evaluation

```python
# pip install torchmetrics
import torchmetrics
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Classification ────────────────────────────────────────────────────────────
acc     = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
f1      = torchmetrics.F1Score(task="multiclass", num_classes=10, average="macro").to(device)
auc     = torchmetrics.AUROC(task="multiclass", num_classes=10).to(device)
prec    = torchmetrics.Precision(task="multiclass", num_classes=10, average="macro").to(device)
confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=10).to(device)

# MetricCollection: apply multiple metrics simultaneously
collection = torchmetrics.MetricCollection({
    "acc":  torchmetrics.Accuracy(task="multiclass", num_classes=10),
    "f1":   torchmetrics.F1Score(task="multiclass", num_classes=10, average="macro"),
    "prec": torchmetrics.Precision(task="multiclass", num_classes=10, average="macro"),
}).to(device)

model.eval()
with torch.inference_mode():
    for x, y in val_loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        collection.update(logits, y)

results = collection.compute()
print({k: v.item() for k, v in results.items()})

# Reset for next epoch
collection.reset()

# ── Detection (Object Detection) ──────────────────────────────────────────────
from torchmetrics.detection.mean_ap import MeanAveragePrecision

map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])

# Predictions: list of dicts with "boxes", "scores", "labels"
preds  = [{"boxes": torch.tensor([[10, 20, 50, 60]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}]
target = [{"boxes": torch.tensor([[12, 18, 52, 58]]), "labels": torch.tensor([0])}]
map_metric.update(preds, target)
print(map_metric.compute())
```

---

## 15.3 Confidence Calibration

A well-calibrated model's confidence reflects its actual accuracy: when it says 80% confident, it should be right 80% of the time.

```python
import numpy as np
import matplotlib.pyplot as plt

def calibration_analysis(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram.
    ECE = Σ (|B_m|/n) · |acc(B_m) - conf(B_m)|
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_probs = probs.max(axis=-1)
    preds     = probs.argmax(axis=-1)
    correct   = (preds == labels).astype(float)

    ece = 0.0
    bin_accs  = []
    bin_confs = []
    bin_sizes = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (max_probs >= lo) & (max_probs < hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_accs.append(0); bin_confs.append(0); bin_sizes.append(0)
            continue

        bin_acc  = correct[mask].mean()
        bin_conf = max_probs[mask].mean()
        ece     += (n_in_bin / len(labels)) * abs(bin_acc - bin_conf)
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_sizes.append(n_in_bin)

    return {"ece": ece, "bin_accs": bin_accs, "bin_confs": bin_confs, "bin_sizes": bin_sizes}


def plot_reliability_diagram(calibration: dict, title: str = "Reliability Diagram") -> plt.Figure:
    """
    Reliability diagram: x-axis = confidence, y-axis = accuracy.
    Perfect calibration = diagonal line.
    """
    n_bins = len(calibration["bin_accs"])
    bin_centers = np.linspace(1/(2*n_bins), 1 - 1/(2*n_bins), n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_centers, calibration["bin_accs"], width=1/n_bins, alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {calibration['ece']:.4f}")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig


# Temperature scaling: post-hoc calibration
class TemperatureScaler(torch.nn.Module):
    """
    Learn a single temperature T that scales logits before softmax.
    Optimise on a validation set (separate from training).
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature.clamp(min=0.05)

    def fit(self, val_loader, device, n_epochs: int = 50):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = torch.nn.CrossEntropyLoss()

        all_logits, all_labels = [], []
        self.model.eval()
        with torch.inference_mode():
            for x, y in val_loader:
                all_logits.append(self.model(x.to(device)).cpu())
                all_labels.append(y)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature.clamp(min=0.05), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Calibrated temperature: {self.temperature.item():.4f}")
```

---

## 15.4 Grad-CAM: Visual Explanations for CNNs

Grad-CAM produces class activation maps by using gradient information flowing into the last convolutional layer.

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).
    Highlights the regions of an image that most influence a prediction.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        # Register hooks to capture activations and gradients
        self._forward_hook  = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Args:
            x: input image tensor (1, C, H, W)
            class_idx: class to explain; if None, uses predicted class
        Returns:
            cam: (H, W) heatmap, values in [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)

        # Forward
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()

        # Backward for target class
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Grad-CAM: α_k^c = (1/Z) Σ_{ij} ∂y^c/∂A_k^{ij}
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalise and resize to input resolution
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


def visualise_gradcam(
    image: torch.Tensor,   # (3, H, W) normalized
    cam: np.ndarray,       # (H, W)
    alpha: float = 0.4,
    title: str = "Grad-CAM",
) -> plt.Figure:
    """Overlay CAM heatmap on the original image."""
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = image.permute(1, 2, 0).numpy() * std + mean
    img  = np.clip(img, 0, 1)

    heatmap = plt.cm.jet(cam)[..., :3]   # (H, W, 3) RGB heatmap
    overlay = alpha * heatmap + (1 - alpha) * img

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);     axes[0].set_title("Original")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title("CAM")
    axes[2].imshow(overlay); axes[2].set_title(title)
    for ax in axes: ax.axis("off")
    return fig


# Usage
model = resnet50(weights="DEFAULT").eval()
cam_extractor = GradCAM(model, target_layer=model.layer4[-1])

x = torch.randn(1, 3, 224, 224)
cam = cam_extractor(x, class_idx=207)   # 207 = golden retriever
cam_extractor.remove_hooks()

fig = visualise_gradcam(x[0], cam, title="Grad-CAM: Golden Retriever")
```

---

## 15.5 Integrated Gradients

IG attributes prediction to input features by integrating gradients along a path from a baseline (usually zeros) to the actual input:

```
IG_i(x) = (xᵢ - x'ᵢ) × ∫₀¹ ∂F(x' + α(x-x')) / ∂xᵢ dα
```

```python
def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,              # (1, C, H, W)
    baseline: torch.Tensor = None, # black image by default
    target_class: int = None,
    n_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Returns attribution map of same shape as x: (1, C, H, W).
    Positive values = features that increase the target class score.
    """
    model.eval()
    x = x.to(device)

    if baseline is None:
        baseline = torch.zeros_like(x)

    # Create interpolated inputs: x' + α(x - x') for α ∈ [0, 1]
    alphas = torch.linspace(0, 1, n_steps, device=device)
    interpolated = baseline + alphas[:, None, None, None, None] * (x - baseline)
    # shape: (n_steps, 1, C, H, W) → reshape to (n_steps, C, H, W)
    interpolated = interpolated.squeeze(1)
    interpolated.requires_grad_(True)

    # Forward and compute gradients
    logits = model(interpolated)
    if target_class is None:
        target_class = logits[-1].argmax().item()

    # Sum target class logits and backprop
    logits[:, target_class].sum().backward()
    grads = interpolated.grad   # (n_steps, C, H, W)

    # Approximate the integral via trapezoidal rule
    avg_grads = grads.mean(dim=0, keepdim=True)   # (1, C, H, W)

    # Attribution = (x - baseline) * avg_grads
    ig = (x - baseline) * avg_grads

    return ig.detach()


# Visualise IG attributions
def plot_ig_attribution(ig: torch.Tensor, image: torch.Tensor) -> plt.Figure:
    """Visualise per-pixel attribution magnitude."""
    # Sum over channels
    attr = ig[0].abs().sum(dim=0).numpy()   # (H, W)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

    img = image[0].permute(1, 2, 0).numpy()
    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(img);     ax1.set_title("Input"); ax1.axis("off")
    ax2.imshow(attr, cmap="hot"); ax2.set_title("Integrated Gradients"); ax2.axis("off")
    return fig
```

---

## 15.6 Attention Visualization for Transformers

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualise_attention(
    model,        # HuggingFace transformer with output_attentions=True
    tokenizer,
    text: str,
    layer: int = -1,
    head: int = 0,
) -> plt.Figure:
    """
    Visualise the attention weights of a transformer layer/head.
    """
    inputs  = tokenizer(text, return_tensors="pt")
    tokens  = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (batch, heads, seq, seq) per layer
    attn = outputs.attentions[layer][0, head].numpy()   # (seq, seq)

    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.5), max(5, len(tokens) * 0.5)))
    im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=attn.max())
    ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens)
    ax.set_title(f"Attention: Layer {layer}, Head {head}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def aggregate_attention(attentions: tuple, aggregate: str = "mean_heads") -> np.ndarray:
    """
    Aggregate multi-layer, multi-head attention into a single matrix.
    Methods: 'mean_heads', 'max_heads', 'attention_rollout'
    """
    # Stack all layers: (n_layers, batch, heads, seq, seq)
    all_attn = torch.stack(attentions)[:, 0]   # (n_layers, heads, seq, seq)

    if aggregate == "mean_heads":
        return all_attn.mean(dim=1).mean(dim=0).numpy()   # mean over layers and heads

    elif aggregate == "attention_rollout":
        # Attention Rollout (Abnar & Zuidema, 2020):
        # Recursively multiply attention maps, adding residual identity
        T = all_attn.shape[-1]
        result = torch.eye(T)
        for layer_attn in all_attn:
            avg_head = layer_attn.mean(dim=0)         # (seq, seq)
            # Add identity for residual connections
            avg_head = avg_head + torch.eye(T)
            # Normalize
            avg_head = avg_head / avg_head.sum(dim=-1, keepdim=True)
            result   = result @ avg_head
        return result.numpy()
```

---

## 15.7 SHAP for Feature Importance

```python
# pip install shap
import shap
import torch
import numpy as np

def shap_explain_tabular(
    model: torch.nn.Module,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Compute SHAP values for a tabular model."""
    model.eval().to(device)

    def predict(x: np.ndarray) -> np.ndarray:
        t = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits = model(t)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    # KernelSHAP: model-agnostic, works with any function
    explainer   = shap.KernelExplainer(predict, X_background)
    shap_values = explainer.shap_values(X_explain, nsamples=100)   # (n_explain, n_features, n_classes)

    # Visualise for first sample, class 0
    shap.summary_plot(shap_values[0], X_explain, feature_names=feature_names, show=False)
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0][0],
        base_values=explainer.expected_value[0],
        data=X_explain[0],
        feature_names=feature_names,
    ), show=False)

    return shap_values


# GradientSHAP for neural networks (faster)
def shap_explain_image(model, x_image, x_background):
    """GradientSHAP for image model."""
    model.eval()
    explainer   = shap.GradientExplainer(model, x_background)
    shap_values = explainer.shap_values(x_image)  # (batch, C, H, W)
    shap.image_plot(shap_values, x_image.permute(0,2,3,1).numpy())
    return shap_values
```

---

## 15.8 Fairness Evaluation

```python
import numpy as np
from typing import Dict

def fairness_audit(
    predictions: np.ndarray,  # model predictions
    labels: np.ndarray,       # ground truth
    groups: np.ndarray,       # sensitive attribute (e.g., gender, race)
    positive_class: int = 1,
) -> Dict:
    """
    Compute fairness metrics across demographic groups.
    Returns: demographic parity, equal opportunity, equalized odds.
    """
    unique_groups = np.unique(groups)
    results = {}

    for g in unique_groups:
        mask = groups == g
        g_preds = predictions[mask]
        g_labels = labels[mask]

        tp = ((g_preds == positive_class) & (g_labels == positive_class)).sum()
        fp = ((g_preds == positive_class) & (g_labels != positive_class)).sum()
        tn = ((g_preds != positive_class) & (g_labels != positive_class)).sum()
        fn = ((g_preds != positive_class) & (g_labels == positive_class)).sum()

        tpr = tp / (tp + fn + 1e-8)   # True Positive Rate (Recall)
        fpr = fp / (fp + tn + 1e-8)   # False Positive Rate
        pr  = (g_preds == positive_class).mean()   # Positive Rate

        results[str(g)] = {"tpr": tpr, "fpr": fpr, "positive_rate": pr, "n_samples": mask.sum()}

    # Fairness gaps
    tprs = [v["tpr"] for v in results.values()]
    fprs = [v["fpr"] for v in results.values()]
    prs  = [v["positive_rate"] for v in results.values()]

    results["fairness_summary"] = {
        "demographic_parity_gap":  max(prs) - min(prs),    # should be ~0
        "equal_opportunity_gap":   max(tprs) - min(tprs),  # TPR gap across groups
        "equalized_odds_gap":      max(max(tprs) - min(tprs), max(fprs) - min(fprs)),
    }
    return results
```

---

## 15.9 Model Card Template

Every production model should ship with a Model Card documenting its purpose, data, performance, and limitations.

```python
def generate_model_card(
    model_name: str,
    model_description: str,
    training_data: dict,
    performance: dict,
    limitations: list,
    intended_use: str,
    out_of_scope_use: str,
) -> str:
    """Generate a standardised model card (Mitchell et al., 2019)."""
    card = f"""
# Model Card: {model_name}

## Model Description
{model_description}

## Intended Use
**Primary intended use:** {intended_use}
**Out-of-scope use:** {out_of_scope_use}

## Training Data
- **Dataset:** {training_data.get('name', 'N/A')}
- **Size:** {training_data.get('size', 'N/A')}
- **Split:** {training_data.get('split', 'N/A')}
- **Preprocessing:** {training_data.get('preprocessing', 'N/A')}

## Performance
| Metric | Value |
|--------|-------|
"""
    for metric, value in performance.items():
        card += f"| {metric} | {value:.4f} |\n"

    card += "\n## Limitations\n"
    for lim in limitations:
        card += f"- {lim}\n"

    card += f"""
## Ethical Considerations
- Fairness audit was performed across gender and age groups
- Model was evaluated for calibration (ECE < 0.05)
- Explainability provided via Grad-CAM / Integrated Gradients

## Model Details
- **Architecture:** See model_config.json
- **Framework:** PyTorch {torch.__version__}
- **License:** MIT
"""
    return card
```

---

## 15.10 End-to-End Evaluation Report

```python
def generate_evaluation_report(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list,
    sample_images: torch.Tensor,
    save_dir: str = "eval_report",
) -> dict:
    """Generate a complete evaluation report with all metrics and plots."""
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. Core metrics
    val_metrics  = evaluate_classifier(model, val_loader,  device, class_names)
    test_metrics = evaluate_classifier(model, test_loader, device, class_names)

    # 2. Calibration
    all_probs, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in test_loader:
            logits = model(x.to(device))
            probs  = torch.softmax(logits, dim=-1).cpu()
            all_probs.append(probs); all_labels.append(y)
    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    calib  = calibration_analysis(probs, labels)
    plot_reliability_diagram(calib).savefig(f"{save_dir}/reliability_diagram.png")

    # 3. Grad-CAM explanations
    cam_extractor = GradCAM(model, target_layer=model.layer4[-1])
    for i, img in enumerate(sample_images[:5]):
        cam = cam_extractor(img.unsqueeze(0), class_idx=labels[i])
        visualise_gradcam(img, cam).savefig(f"{save_dir}/gradcam_{i}.png")
    cam_extractor.remove_hooks()

    # 4. Confusion matrix
    cm = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, class_names).savefig(f"{save_dir}/confusion_matrix.png")

    return {
        "val":   val_metrics,
        "test":  test_metrics,
        "ece":   calib["ece"],
        "saved_to": save_dir,
    }
```

---

## Exercises

**Exercise 15.1** Evaluate a ResNet-50 on ImageNet-1K validation set using `torchmetrics`. Report Top-1 accuracy, Top-5 accuracy, macro F1, and the 10 classes with the lowest recall.

**Exercise 15.2** Apply Grad-CAM to a misclassified image from CIFAR-10. Does the heatmap reveal what confused the model? Compare Grad-CAM vs GradCAM++ vs SmoothGrad explanations.

**Exercise 15.3** Apply temperature scaling to a trained CIFAR-10 classifier. Plot the reliability diagram before and after calibration. Report ECE improvement.

---

## Module Summary

| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| Accuracy/F1/AUC | Overall performance | Always |
| Confusion matrix | Per-class errors | Classification |
| BLEU/ROUGE/BERTScore | Sequence quality | NLP generation |
| ECE + reliability diagram | Probability calibration | Safety-critical |
| Grad-CAM | CNN spatial attribution | Image models |
| Integrated Gradients | Feature-level attribution | Any differentiable model |
| Attention visualization | What tokens model attends to | Transformers |
| SHAP | Model-agnostic attribution | Any model, tabular data |
| Fairness audit | Demographic equity | Regulated industries |

---

## Final Quiz

1. What is ECE (Expected Calibration Error) and what does a value of 0.05 mean?
2. How does Grad-CAM differ from Integrated Gradients in its approach to attribution?
3. What is temperature scaling and does it change model accuracy?
4. What is Attention Rollout and how does it improve over raw attention weights?
5. What is demographic parity and when should it be prioritised over accuracy?
6. Why should you evaluate on a *test* set held out from calibration?
7. What does a BLEU score of 40 mean for machine translation?

---

*Congratulations on completing the PyTorch course!*
*Proceed to [Capstone Projects & Solutions](./PROJECTS_AND_SOLUTIONS.md)*
