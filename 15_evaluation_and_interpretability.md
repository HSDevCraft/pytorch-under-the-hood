# Module 15: Evaluation & Interpretability — Understanding What Your Model Learned

> **Goal:** Go beyond accuracy — measure model quality rigorously, diagnose failures, explain predictions, and ensure fairness before deployment.

---

## Learning Objectives

By the end of this module, you will:
- **Compute** and interpret evaluation metrics for classification, regression, and NLP
- **Use** torchmetrics for scalable, multi-GPU-compatible metrics
- **Measure** and fix model calibration (confidence ≠ accuracy is a problem)
- **Visualize** what a model learned using Grad-CAM
- **Explain** individual predictions with Integrated Gradients
- **Audit** your model for demographic fairness
- **Write** a model card for responsible deployment

---

## Part 1: Evaluation Metrics Deep Dive

### 1.1 Beyond Accuracy — Why It's Not Enough

```python
# Scenario: Fraud detection dataset
# 99% legitimate transactions, 1% fraud
# A model that always predicts "legitimate" achieves 99% accuracy
# but catches ZERO fraud cases — completely useless!

# Better metrics:
# Precision: of all "fraud" predictions, how many are actually fraud?
# Recall:    of all actual frauds, how many did we catch?
# F1:        harmonic mean of precision and recall

import torch
import torchmetrics

n_classes = 2  # Binary: fraud or not

# Create metrics
accuracy   = torchmetrics.Accuracy(task='binary')
precision  = torchmetrics.Precision(task='binary')
recall     = torchmetrics.Recall(task='binary')
f1         = torchmetrics.F1Score(task='binary')
auroc      = torchmetrics.AUROC(task='binary')         # Area Under ROC Curve
avg_prec   = torchmetrics.AveragePrecision(task='binary')  # PR curve

# Simulate predictions
torch.manual_seed(42)
preds  = torch.tensor([0.9, 0.05, 0.8, 0.3, 0.95, 0.1])  # Probabilities
labels = torch.tensor([1, 0, 1, 0, 1, 0])                  # True labels

# Compute all metrics
print(f"Accuracy:   {accuracy(preds, labels):.4f}")
print(f"Precision:  {precision(preds, labels):.4f}")  # TP / (TP + FP)
print(f"Recall:     {recall(preds, labels):.4f}")     # TP / (TP + FN)
print(f"F1 Score:   {f1(preds, labels):.4f}")
print(f"AUROC:      {auroc(preds, labels):.4f}")      # 1.0 = perfect separator

# Confusion matrix — the most informative single view
conf_matrix = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)
cm = conf_matrix(preds, labels)
print(f"\nConfusion Matrix:\n{cm}")
# [[TN, FP],
#  [FN, TP]]
```

### 1.2 Multi-class Metrics

```python
# For multi-class classification, averaging strategy matters:
# - 'macro':   average across classes equally (treats all classes same)
#              Use when: all classes are equally important
# - 'micro':   aggregate predictions globally (dominated by large classes)
#              Use when: class imbalance exists
# - 'weighted': weight by class frequency
#              Use when: imbalanced but want frequency-aware average

n_classes = 10  # CIFAR-10

logits = torch.randn(100, 10)   # Raw model outputs
preds  = logits.argmax(dim=1)   # Predicted classes
labels = torch.randint(0, 10, (100,))  # True labels

# Macro F1 — treats all 10 classes equally
f1_macro = torchmetrics.F1Score(task='multiclass', num_classes=10, average='macro')
print(f"Macro F1:    {f1_macro(preds, labels):.4f}")

# Top-5 accuracy — is correct class in top-5 predictions? (ImageNet standard)
top5_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10, top_k=5)
print(f"Top-5 Acc:   {top5_acc(logits, labels):.4f}")
```

### 1.3 Regression Metrics

```python
preds_reg  = torch.randn(100)
targets_reg = torch.randn(100)

mse  = torchmetrics.MeanSquaredError()
mae  = torchmetrics.MeanAbsoluteError()
r2   = torchmetrics.R2Score()      # Coefficient of determination

rmse = torch.sqrt(mse(preds_reg, targets_reg))
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae(preds_reg, targets_reg):.4f}")
print(f"R²:   {r2(preds_reg, targets_reg):.4f}")
# R² = 1.0: perfect fit, R² = 0: as good as predicting mean, R² < 0: worse than mean
```

### 1.4 NLP Metrics

```python
# BLEU: measures n-gram overlap for machine translation
# Range: 0 (no overlap) to 100 (perfect match)
from torchmetrics.text import BLEUScore

bleu = BLEUScore(n_gram=4)
hypotheses = ["the cat is on the mat"]    # Model output
references  = [["the cat is on the mat"]] # Ground truth

bleu_score = bleu(hypotheses, references)
print(f"BLEU score: {bleu_score:.4f}")

# Perplexity for language models
# Lower = better (model is less "surprised" by the text)
# Formula: exp(average negative log-likelihood)
cross_entropy_loss = 3.5  # Example: NLL per token
perplexity = torch.exp(torch.tensor(cross_entropy_loss))
print(f"Perplexity: {perplexity:.1f}")  # ~33 for language models
```

---

## Part 2: Confidence Calibration

### 2.1 The Calibration Problem

A well-calibrated model: when it says "70% confident," it's correct 70% of the time.
An overconfident model: says "99% confident" but is only right 70% of the time.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE).
    
    Algorithm:
    1. Divide predictions into n_bins confidence buckets
    2. In each bucket: compute average confidence and average accuracy
    3. ECE = weighted average of |confidence - accuracy| across buckets
    
    ECE = Σ (|B_m| / n) × |acc(B_m) - conf(B_m)|
    
    Interpretation:
    ECE = 0.05: on average, confidence is 5 percentage points off from accuracy
    ECE < 0.02: excellent calibration (top production models)
    ECE > 0.10: significant calibration problem
    """
    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels).float()
    
    ece = 0.0
    n_total = len(labels)
    
    for bin_lower, bin_upper in zip(
        np.linspace(0, 1, n_bins + 1)[:-1],
        np.linspace(0, 1, n_bins + 1)[1:]
    ):
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        n_in_bin = in_bin.sum().item()
        
        if n_in_bin > 0:
            avg_confidence = confidences[in_bin].mean().item()
            avg_accuracy   = correct[in_bin].mean().item()
            ece += (n_in_bin / n_total) * abs(avg_confidence - avg_accuracy)
    
    return ece


def plot_reliability_diagram(probs: torch.Tensor, labels: torch.Tensor,
                              n_bins: int = 10, title: str = "Reliability Diagram"):
    """
    Reliability diagram visualizes calibration.
    
    Perfect calibration: all points on the diagonal y=x
    Overconfidence: points below diagonal (confidence > accuracy)
    Underconfidence: points above diagonal (accuracy > confidence)
    """
    confidences, predictions = probs.max(dim=1)
    correct = (predictions == labels).float()
    
    bin_accs, bin_confs = [], []
    for b in range(n_bins):
        low = b / n_bins
        high = (b + 1) / n_bins
        in_bin = (confidences >= low) & (confidences < high)
        if in_bin.sum() > 0:
            bin_accs.append(correct[in_bin].mean().item())
            bin_confs.append(confidences[in_bin].mean().item())
    
    ece = compute_ece(probs, labels, n_bins)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.bar(bin_confs, bin_accs, width=0.1, alpha=0.7, label='Model')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"{title}\nECE = {ece:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reliability_diagram.png")
    plt.show()
    return ece
```

### 2.2 Temperature Scaling — Simple, Effective Calibration Fix

```python
class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration: learn a single scalar T on validation set.
    
    calibrated_prob = softmax(logits / T)
    
    T > 1: softer probabilities (fixes overconfidence)
    T < 1: sharper probabilities (fixes underconfidence)
    T = 1: original model (no change)
    
    Key property: does NOT change predictions (argmax is same)!
    Only changes confidence values → no accuracy impact.
    """
    
    def __init__(self):
        super().__init__()
        # Initialize T=1 (no change at start)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)  # Prevent T→0


def calibrate_model(model: nn.Module, val_loader, device: str = 'cpu') -> TemperatureScaling:
    """
    Fit temperature on validation set using negative log-likelihood.
    Optimization: minimize NLL(calibrated_logits) on val set.
    """
    # Collect all validation logits and labels
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_logits.append(logits)
            all_labels.append(y)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Fit temperature
    temp_model = TemperatureScaling()
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=100)
    nll_criterion = nn.CrossEntropyLoss()
    
    def eval_fn():
        optimizer.zero_grad()
        scaled_logits = temp_model(all_logits)
        loss = nll_criterion(scaled_logits, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_fn)
    
    T = temp_model.temperature.item()
    print(f"Fitted temperature: T = {T:.4f}")
    
    # Compare ECE before and after
    probs_before = torch.softmax(all_logits, dim=1)
    probs_after  = torch.softmax(temp_model(all_logits), dim=1)
    
    ece_before = compute_ece(probs_before, all_labels)
    ece_after  = compute_ece(probs_after, all_labels)
    
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after  calibration: {ece_after:.4f}")
    print(f"ECE reduction: {(ece_before - ece_after) / ece_before:.1%}")
    
    return temp_model
```

---

## Part 3: Grad-CAM — Visualizing What the Model Sees

### 3.1 The Intuition

Grad-CAM answers: "Which **spatial regions** were most important for this prediction?"

It uses the gradient of the output class score with respect to the final convolutional feature maps. High gradient + high activation = important region.

```python
import torch
import torch.nn as nn
import numpy as np
import cv2

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Algorithm:
    1. Forward pass → get class score
    2. Backward pass → get gradients w.r.t. target conv layer
    3. Global average pool gradients → get importance weights α_k
    4. Weighted combination of feature maps + ReLU → CAM
    5. Upsample CAM to input resolution
    
    Formula: CAM = ReLU( Σ_k α_k * A_k )
    where α_k = (1/Z) Σ_ij (∂y^c / ∂A_k_ij)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        model: the full CNN model
        target_layer: the conv layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Hooks to capture feature maps and gradients
        self._feature_maps = None
        self._gradients = None
        
        # Register forward hook: captures feature maps during forward pass
        self._fwd_hook = target_layer.register_forward_hook(self._save_feature_maps)
        # Register backward hook: captures gradients during backward pass
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)
    
    def _save_feature_maps(self, module, input, output):
        """Saves the feature maps from the forward pass"""
        self._feature_maps = output.detach()  # Shape: (batch, channels, H, W)
    
    def _save_gradients(self, module, grad_input, grad_output):
        """Saves the gradients from the backward pass"""
        self._gradients = grad_output[0].detach()  # Shape: (batch, channels, H, W)
    
    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate CAM for the given input.
        
        class_idx: if None, uses the predicted class
        Returns: CAM heatmap normalized to [0, 1], shape (H, W)
        """
        self.model.eval()
        
        # Step 1: Forward pass (hooks save feature maps)
        logits = self.model(x)                    # (1, n_classes)
        
        # Default: visualize the top predicted class
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        # Step 2: Backward pass (hooks save gradients)
        self.model.zero_grad()
        # Create a score for only the target class
        class_score = logits[0, class_idx]
        class_score.backward()
        
        # Step 3: Compute importance weights α_k
        # Global average pool of gradients over spatial dimensions
        gradients   = self._gradients[0]       # (channels, H, W)
        feature_maps = self._feature_maps[0]   # (channels, H, W)
        
        # α_k: importance weight for channel k
        alpha = gradients.mean(dim=(1, 2))     # (channels,) — one weight per channel
        
        # Step 4: Weighted combination → CAM
        # cam[i,j] = Σ_k α_k * A_k[i,j]
        cam = torch.zeros(feature_maps.shape[1:], device=x.device)  # (H, W)
        for k, a in enumerate(alpha):
            cam += a * feature_maps[k]
        
        # Step 5: ReLU (focus on regions that positively influence class score)
        cam = torch.relu(cam)
        
        # Step 6: Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def overlay_on_image(self, image: np.ndarray, cam: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
        """
        Superimpose heatmap on original image.
        
        image: HxWx3 RGB image (values in [0, 255] or [0, 1])
        cam: HxW heatmap (values in [0, 1])
        """
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Apply colormap (jet: blue=low importance, red=high importance)
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Normalize image to [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Overlay
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        return overlay
    
    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks"""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# Usage example:
# model = torchvision.models.resnet50(pretrained=True).eval()
# grad_cam = GradCAM(model, target_layer=model.layer4[-1].conv2)
# 
# image = load_image("cat.jpg")  # (1, 3, 224, 224) tensor
# cam = grad_cam(image, class_idx=281)  # 281 = tabby cat in ImageNet
# 
# overlay = grad_cam.overlay_on_image(image_np, cam)
# plt.imshow(overlay); plt.title("Grad-CAM: What the model sees"); plt.show()
# grad_cam.remove_hooks()
```

---

## Part 4: Integrated Gradients — Feature Attribution

### 4.1 Why Integrated Gradients

Grad-CAM only works for CNNs (spatial feature maps). **Integrated Gradients** works for any model (tabular, text, images) and attributes importance to individual input features.

```python
def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,           # Input to explain: (1, *input_shape)
    baseline: torch.Tensor,    # Baseline input (usually zeros): same shape as x
    target_class: int,         # Class to explain
    n_steps: int = 300,        # Number of interpolation steps (more = accurate)
) -> torch.Tensor:
    """
    Integrated Gradients computes:
    IG(x, x') = (x - x') × ∫₀¹ ∂F(x' + α(x-x'))/∂x dα
    
    Where x' is the baseline (e.g., black image or average image).
    
    Intuition: Measure how much each feature contributes by watching
    how the prediction changes as we "add" each feature from zero to full.
    
    Properties:
    - Completeness: IG attributions sum to F(x) - F(x')
    - Sensitivity: non-zero attribution iff varying that feature changes prediction
    - Implementation invariance: equivalent networks get same attributions
    """
    model.eval()
    
    # Create interpolated inputs: x' + α*(x - x') for α in [0, 1]
    # α=0: pure baseline, α=1: original input
    alphas = torch.linspace(0, 1, n_steps)  # (n_steps,)
    
    # Interpolated inputs: (n_steps, *input_shape)
    interpolated = baseline + alphas.view(n_steps, *([1] * (x.dim() - 1))) * (x - baseline)
    interpolated.requires_grad_(True)
    
    # Compute gradients at each interpolation step
    logits = model(interpolated)  # (n_steps, n_classes)
    
    # Backpropagate score for target class
    target_score = logits[:, target_class].sum()
    target_score.backward()
    
    grads = interpolated.grad  # (n_steps, *input_shape)
    
    # Integrate gradients using trapezoidal rule (average = approximation)
    avg_grads = grads.mean(dim=0)  # Average over interpolation steps
    
    # Multiply by (x - baseline) — the "path" from baseline to input
    integrated_grads = (x.squeeze(0) - baseline.squeeze(0)) * avg_grads
    
    return integrated_grads.detach()


# Visualize attributions for an image classifier
def visualize_attributions(image: torch.Tensor, attributions: torch.Tensor):
    """Show which pixels contributed most to the prediction"""
    # Attribution magnitude: sum absolute values across color channels
    attr_magnitude = attributions.abs().sum(dim=0)  # (H, W)
    
    # Normalize
    attr_magnitude = (attr_magnitude - attr_magnitude.min())
    attr_magnitude = attr_magnitude / (attr_magnitude.max() + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    im = axes[1].imshow(attr_magnitude.cpu().numpy(), cmap='hot')
    axes[1].set_title("Integrated Gradients Attribution\n(Red = High Importance)")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("integrated_gradients.png")
    plt.show()
```

---

## Part 5: Fairness Audit

### 5.1 Detecting Demographic Bias

```python
from typing import Dict

def fairness_audit(
    model: nn.Module,
    dataset,
    demographic_attribute: str,  # e.g., 'gender', 'race', 'age_group'
    device: str = 'cpu'
) -> Dict[str, Dict]:
    """
    Measure model performance across demographic groups.
    
    Key fairness metrics:
    - Demographic Parity: P(ŷ=1 | group=A) = P(ŷ=1 | group=B)
      Equal positive prediction rates across groups
    - Equal Opportunity:  TPR_A = TPR_B
      Equal true positive rates (recall)
    - Equalized Odds:     TPR_A = TPR_B AND FPR_A = FPR_B
      Equal error rates across groups
    
    None of these are perfectly achievable simultaneously (impossibility theorem)
    Choose based on what matters most for your application.
    """
    model.eval()
    
    # Group data by demographic attribute
    groups = {}
    for sample in dataset:
        group = sample[demographic_attribute]
        if group not in groups:
            groups[group] = {'preds': [], 'labels': []}
        
        x = sample['features'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1).item()
        
        groups[group]['preds'].append(pred)
        groups[group]['labels'].append(sample['label'])
    
    # Compute metrics per group
    results = {}
    for group_name, data in groups.items():
        preds  = torch.tensor(data['preds'])
        labels = torch.tensor(data['labels'])
        
        accuracy  = (preds == labels).float().mean()
        pos_rate  = (preds == 1).float().mean()  # Positive prediction rate
        
        # True Positive Rate (recall) — crucial for high-stakes decisions
        true_positives = ((preds == 1) & (labels == 1)).float().sum()
        actual_positives = (labels == 1).float().sum()
        tpr = true_positives / (actual_positives + 1e-8)
        
        # False Positive Rate
        false_positives = ((preds == 1) & (labels == 0)).float().sum()
        actual_negatives = (labels == 0).float().sum()
        fpr = false_positives / (actual_negatives + 1e-8)
        
        results[group_name] = {
            'n_samples': len(labels),
            'accuracy':  accuracy.item(),
            'pos_rate':  pos_rate.item(),    # For demographic parity
            'tpr':       tpr.item(),          # For equal opportunity
            'fpr':       fpr.item(),          # For equalized odds
        }
    
    # Compute parity gaps
    group_names = list(results.keys())
    parity_gaps = {}
    for metric in ['accuracy', 'pos_rate', 'tpr', 'fpr']:
        values = [results[g][metric] for g in group_names]
        parity_gaps[f'{metric}_gap'] = max(values) - min(values)
    
    print("\n=== Fairness Audit Results ===")
    for group, metrics in results.items():
        print(f"\n{group} (n={metrics['n_samples']}):")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Positive rate:     {metrics['pos_rate']:.4f}")
        print(f"  True positive rate:{metrics['tpr']:.4f}")
        print(f"  False positive rate:{metrics['fpr']:.4f}")
    
    print("\n=== Parity Gaps (max-min across groups) ===")
    for metric, gap in parity_gaps.items():
        status = "⚠ CONCERN" if gap > 0.05 else "✓ OK"
        print(f"  {metric}: {gap:.4f} {status}")
    
    return results, parity_gaps
```

---

## Part 6: Model Card

### 6.1 Responsible AI Documentation

```python
def generate_model_card(
    model_name: str,
    task: str,
    training_data: dict,
    performance: dict,
    fairness: dict,
    limitations: list,
    intended_use: str,
    out_of_scope: list,
) -> str:
    """
    Generate a structured model card for responsible deployment.
    
    Model cards are documentation artifacts that:
    - Describe what the model does and doesn't do
    - Report performance across different groups
    - Acknowledge limitations and potential harms
    - Enable informed decisions by downstream users
    """
    
    card = f"""# Model Card: {model_name}

## Model Description
- **Task:** {task}
- **Architecture:** [describe here]
- **Framework:** PyTorch 2.0+

## Intended Use
{intended_use}

## Out-of-Scope Uses
{chr(10).join(f"- {s}" for s in out_of_scope)}

## Training Data
- **Source:** {training_data.get('source', 'N/A')}
- **Size:** {training_data.get('size', 'N/A')}
- **Date range:** {training_data.get('date_range', 'N/A')}
- **Known biases:** {training_data.get('known_biases', 'None documented')}

## Performance

### Overall
{chr(10).join(f"- **{k}:** {v}" for k, v in performance.items())}

### Fairness Across Groups
{chr(10).join(f"- **{k}:** {v}" for k, v in fairness.items())}

## Limitations
{chr(10).join(f"- {l}" for l in limitations)}

## Ethical Considerations
This model should not be used for decisions involving [enumerate high-risk use cases].
Performance may degrade for inputs from distributions not represented in training data.

## Citation
[Add citation if applicable]
"""
    return card


# Example model card
card = generate_model_card(
    model_name="FraudDetector-v2.1",
    task="Binary classification: fraudulent vs. legitimate transactions",
    training_data={
        'source': 'Internal transaction log 2020-2023',
        'size': '50M transactions',
        'date_range': '2020-01-01 to 2023-12-31',
        'known_biases': 'Under-represented: rural customers, new account holders',
    },
    performance={
        'AUROC (overall)': 0.971,
        'Precision@0.5': 0.843,
        'Recall@0.5': 0.891,
        'F1@0.5': 0.866,
    },
    fairness={
        'Recall gap (by region)': '0.023 — acceptable',
        'FPR gap (by age group)': '0.018 — acceptable',
        'FPR gap (by account age)': '0.041 — monitoring required',
    },
    limitations=[
        "Performance degrades for transaction amounts >$50,000 (rare in training data)",
        "Not validated on non-US payment networks",
        "Requires retraining every 6 months as fraud patterns evolve",
    ],
    intended_use="Assist fraud analysts; not for fully automated transaction blocking without human review",
    out_of_scope=["Credit scoring", "Customer creditworthiness assessment", "Identity verification"],
)
print(card)
```

---

## Key Takeaways

| Concept | What It Reveals | Tool/Method |
|---------|----------------|-------------|
| **Confusion matrix** | Per-class error patterns | torchmetrics.ConfusionMatrix |
| **AUROC** | Overall discriminative ability | torchmetrics.AUROC |
| **ECE** | Confidence reliability | Custom ECE function |
| **Temperature scaling** | Fix overconfidence | Fit T on validation set |
| **Grad-CAM** | Spatial attention heatmap | Gradient × feature map |
| **Integrated Gradients** | Feature attribution | Path integral of gradients |
| **Fairness audit** | Demographic disparities | Per-group metric breakdown |
| **Model card** | Responsible disclosure | Structured documentation |

---

## Quiz

1. **Why is accuracy insufficient for imbalanced datasets?**
   - Answer: A model predicting the majority class always achieves high accuracy while completely ignoring the minority class

2. **What does AUROC of 1.0 mean?**
   - Answer: Perfect discrimination — model ranks all positive examples above all negatives regardless of threshold

3. **What is ECE and what does ECE=0.05 mean?**
   - Answer: Expected Calibration Error; model's confidence is on average 5 percentage points off from its actual accuracy

4. **Does temperature scaling change model predictions?**
   - Answer: No — it only scales logits, leaving the argmax unchanged; only confidence values (probabilities) change

5. **What is Grad-CAM measuring?**
   - Answer: Which spatial regions in the input caused the highest gradient in the output score, weighted by feature map activation

6. **What is the baseline in Integrated Gradients?**
   - Answer: A reference input (often zeros or average image) representing "absence" of information

7. **What is demographic parity?**
   - Answer: Equal positive prediction rates across demographic groups

8. **Why are there multiple fairness metrics and why can't all be satisfied?**
   - Answer: Accuracy, demographic parity, and equalized odds are mathematically incompatible (Chouldechova's impossibility theorem); must choose based on context

9. **What does Grad-CAM's `register_forward_hook` do?**
   - Answer: Registers a callback that captures the layer's output (feature maps) during the forward pass

10. **What is a model card and why is it important?**
    - Answer: Structured documentation describing a model's intended use, performance across groups, and limitations; enables informed deployment decisions and responsible AI practices
