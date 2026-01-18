# Domain Feature Usage - Integration Guide

## Problem
You're unsure if the model is actually using domain features (InterPro/Signature IDs) effectively, even though they're being concatenated with ESM embeddings.

## Solutions Provided

### 🔍 **Solution 1: Quick Diagnostic Check** (Recommended First Step)

Add this code **after training** each aspect model to verify domain usage:

```python
# Add this import at the top of your script
from domain_diagnostics import run_all_diagnostics

# After training each aspect (inside train_all function, after train_model):
print(f"\n{'='*80}")
print(f"DIAGNOSING DOMAIN FEATURE USAGE FOR {aspect}")
print(f"{'='*80}")

diagnostics = run_all_diagnostics(model, val_loader, device)

# If domains aren't being used effectively, you'll see warnings here
```

**What this tells you:**
- ✅ **Gradient importance**: How much the model relies on domains vs ESM
- ✅ **Ablation test**: How predictions change when domains are removed
- ✅ **Clear warnings**: If domains have <0.1% impact, you'll get alerts

---

### 🎯 **Solution 2: Use Improved Model with Explicit Attention** (Best Fix)

Replace your current model with the improved version that explicitly tracks feature usage:

```python
# In your main script, replace the import:
from domain_diagnostics import ImprovedProteinClassifierWithDomains, FeatureUsageMonitor

# In train_all(), replace model instantiation:
model = ImprovedProteinClassifierWithDomains(  # Changed from ProteinClassifierWithDomains
    EMBED_DIM, num_classes, config['hidden_dims'], config['dropout'],
    num_domains=NUM_DOMAINS, domain_output_dim=CONFIG['domain_output_dim'],
    use_domains=USE_DOMAINS
)

# Add feature monitoring:
feature_monitor = FeatureUsageMonitor()

# Inside the training loop, after each epoch:
# (Add this after computing validation metrics)
feature_monitor.log(epoch, model)

# Print feature usage stats
if epoch % 5 == 0:  # Every 5 epochs
    weights = model.get_feature_weights()
    if weights:
        print(f"    Feature weights - ESM: {weights['esm_weight']:.3f}, "
              f"Domain: {weights['domain_weight']:.3f}")

# After training completes:
feature_monitor.plot(aspect)  # Creates visualization
```

**What this gives you:**
- ✅ **Attention weights**: See exactly how much the model uses ESM vs domains
- ✅ **Training visualization**: Plot showing feature usage over time
- ✅ **Real-time monitoring**: Track feature weights every epoch

---

### 📊 **Solution 3: Add Per-Domain Analysis**

Find out **which specific domains** are most important:

```python
from domain_diagnostics import analyze_domain_contributions

# After training, create domain vocabulary mapping:
# (You'll need to extract this from your domain loading code)
domain_to_idx = {}  # Your existing mapping from load_domain_features
idx_to_domain = {idx: domain for domain, idx in domain_to_idx.items()}

# Analyze which domains matter most:
domain_importance = analyze_domain_contributions(
    model, val_loader, domain_to_idx, idx_to_domain, device, top_k=20
)
```

---

## 🚀 **Quick Integration Example**

Here's the minimal code to add to your existing script:

```python
# At the top, add import:
from domain_diagnostics import (
    run_all_diagnostics,
    ImprovedProteinClassifierWithDomains,
    FeatureUsageMonitor
)

# In train_all(), after model training for each aspect:
def train_all():
    # ... existing code ...

    for aspect in ASPECTS:
        # ... existing training code ...

        model, fmax, threshold = train_model(aspect, model, train_loader, val_loader,
                                              ia_weights, config, class_weights)

        # ===== ADD THIS DIAGNOSTIC CODE =====
        print(f"\n🔍 Checking domain feature usage for {aspect}...")
        diagnostics = run_all_diagnostics(model, val_loader, device)

        # Log results
        results.append({
            **results[-1],  # existing results
            'domain_impact': diagnostics['ablation']['mean_diff'],
            'domain_used': diagnostics['ablation']['mean_diff'] > 0.01
        })
        # ===== END DIAGNOSTIC CODE =====
```

---

## 📈 **Expected Results**

### ✅ **Good Domain Usage:**
```
ABLATION TEST (With vs Without Domains)
========================================================
Mean absolute difference:    0.0234
Max absolute difference:     0.1567
Predictions changed >1%:     45.23%

Interpretation:
  ✓  GOOD: Domains have meaningful impact
```

### ⚠️ **Poor Domain Usage (Problem!):**
```
ABLATION TEST (With vs Without Domains)
========================================================
Mean absolute difference:    0.0003
Max absolute difference:     0.0089
Predictions changed >1%:     2.34%

Interpretation:
  ⚠️  WARNING: Domains have minimal impact (<0.1%)
      Model may not be using domain features!
```

---

## 🔧 **If Domains Aren't Being Used**

### Fix #1: Use Explicit Feature Fusion
```python
# Replace your model with ImprovedProteinClassifierWithDomains
# This uses separate pathways with attention, forcing the model to consider domains
```

### Fix #2: Increase Domain Influence
```python
CONFIG = {
    # ... existing config ...
    'domain_output_dim': 512,  # Increase from 256
    'domain_embed_dim': 128,   # Increase from 64
}
```

### Fix #3: Add Domain-Specific Loss
```python
# Add auxiliary loss that forces model to predict something from domains alone
class DomainAwareLoss(nn.Module):
    def forward(self, outputs, labels, domain_features):
        main_loss = asymmetric_loss(outputs, labels)

        # Force model to make reasonable predictions from domains alone
        domain_only_outputs = self.domain_head(domain_features)
        domain_loss = asymmetric_loss(domain_only_outputs, labels)

        return main_loss + 0.3 * domain_loss  # Weighted combination
```

### Fix #4: Separate Learning Rates
```python
# Give domain encoder higher learning rate
optimizer = optim.AdamW([
    {'params': model.esm_pathway.parameters(), 'lr': 5e-4},
    {'params': model.domain_encoder.parameters(), 'lr': 1e-3},  # Higher LR
    {'params': model.fusion.parameters(), 'lr': 7e-4}
], weight_decay=CONFIG['weight_decay'])
```

---

## 📝 **Summary Checklist**

- [ ] Run `run_all_diagnostics()` on your current model
- [ ] Check if mean_diff > 0.01 (domains are being used)
- [ ] If not, switch to `ImprovedProteinClassifierWithDomains`
- [ ] Add `FeatureUsageMonitor` to track feature weights during training
- [ ] Review per-domain importance to see which domains help most
- [ ] Adjust hyperparameters if needed (domain_output_dim, LR, etc.)

---

## 🎯 **Bottom Line**

**Your current model concatenates features, but that doesn't guarantee they're used!**

The diagnostic tools will:
1. ✅ Tell you if domains are actually being used
2. ✅ Show you HOW MUCH they're being used
3. ✅ Identify WHICH domains are most important
4. ✅ Provide fixes if they're being ignored

Run the diagnostics first, then apply fixes based on what you find.
