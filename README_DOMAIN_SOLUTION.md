# Solution: How to Verify Domain Features Are Being Used

## 🎯 Your Problem

You're using ESM embeddings + InterPro/Signature domain IDs in your model, but **you're not sure if the model is actually using the domain information** or just relying on ESM embeddings.

### Why This Matters

Just because you concatenate domain features with ESM embeddings doesn't mean the model uses them! The model might:
- ❌ Ignore domain features completely
- ❌ Give them very low weight (< 1% impact)
- ❌ Only use ESM embeddings to make predictions

**You need to verify and ensure domain features are being utilized.**

---

## ✅ The Solution (3 Steps)

### Step 1: Run Diagnostics (5 minutes)

**Quick test to see if domains are being used:**

```python
# Add to your training script after training each model:
from domain_diagnostics import ablation_test

# After: model, fmax, threshold = train_model(...)
ablation_test(model, val_loader, device)
```

**What this tells you:**
- Mean difference > 0.01 → ✅ Domains are being used
- Mean difference < 0.01 → ⚠️ Domains have weak impact
- Mean difference < 0.001 → ❌ Domains are NOT being used

---

### Step 2: Use Improved Model (if needed)

**If diagnostics show domains aren't being used, switch to the improved model:**

```python
from domain_diagnostics import ImprovedProteinClassifierWithDomains

# Replace your current model with:
model = ImprovedProteinClassifierWithDomains(
    esm_dim=EMBED_DIM,
    num_classes=num_classes,
    hidden_dims=config['hidden_dims'],
    dropout=config['dropout'],
    num_domains=NUM_DOMAINS,
    domain_output_dim=CONFIG['domain_output_dim'],
    use_domains=USE_DOMAINS
)
```

**What's different:**
- ✅ Separate pathways for ESM vs Domain features
- ✅ Explicit attention mechanism showing feature weights
- ✅ Forces model to consider both feature types
- ✅ You can see ESM/Domain weights in real-time

---

### Step 3: Monitor During Training

**Track feature usage throughout training:**

```python
from domain_diagnostics import FeatureUsageMonitor

# In train_model():
feature_monitor = FeatureUsageMonitor()

# In training loop:
for epoch in range(epochs):
    # ... training code ...

    # Log feature weights
    feature_monitor.log(epoch, model)

    # Print every 5 epochs
    if epoch % 5 == 0:
        weights = model.get_feature_weights()
        if weights:
            print(f"ESM: {weights['esm_weight']:.3f}, Domain: {weights['domain_weight']:.3f}")

# After training:
feature_monitor.plot(aspect)  # Creates visualization
```

**What you'll see:**
```
Ep  5 | ... | Fmax 0.4123 | ...
    💡 Feature weights - ESM: 0.687, Domain: 0.313

Ep 10 | ... | Fmax 0.4567 | ...
    💡 Feature weights - ESM: 0.702, Domain: 0.298
```

---

## 📁 Files Provided

| File | Purpose |
|------|---------|
| `domain_diagnostics.py` | Core diagnostic tools and improved model |
| `test_domain_usage.py` | Standalone test script |
| `DOMAIN_FEATURE_INTEGRATION_GUIDE.md` | Detailed integration guide |
| `CHANGES_TO_MAIN_SCRIPT.md` | Exact code changes to make |
| `README_DOMAIN_SOLUTION.md` | This file (quick start) |

---

## 🚀 Quick Start (Choose One)

### Option A: Just Check (1 line of code)

```python
from domain_diagnostics import ablation_test
ablation_test(model, val_loader, device)  # Add after training
```

### Option B: Full Solution (Better)

1. Replace model class → `ImprovedProteinClassifierWithDomains`
2. Add feature monitoring → `FeatureUsageMonitor`
3. Check diagnostics → `ablation_test()`

See `CHANGES_TO_MAIN_SCRIPT.md` for exact code.

---

## 🔍 What Each Tool Does

### `ablation_test(model, val_loader, device)`
- **What**: Compares predictions WITH vs WITHOUT domain features
- **When**: After training, to verify domain impact
- **Output**: Mean prediction difference (higher = domains used more)

### `compute_feature_importance(model, val_loader, device)`
- **What**: Measures gradient-based importance of ESM vs Domain features
- **When**: After training, for detailed analysis
- **Output**: Importance scores and ratio

### `ImprovedProteinClassifierWithDomains`
- **What**: Model with explicit ESM/Domain pathways and attention
- **When**: Use instead of `ProteinClassifierWithDomains` if domains aren't being used
- **Output**: Same predictions, but with visible feature weights

### `FeatureUsageMonitor`
- **What**: Tracks feature weights throughout training
- **When**: During training, to monitor domain feature usage
- **Output**: Plots and statistics showing feature weight evolution

### `run_all_diagnostics(model, val_loader, device)`
- **What**: Runs all diagnostic tests at once
- **When**: After training, for comprehensive analysis
- **Output**: Complete report with recommendations

---

## 📊 Example Output

### Good Domain Usage ✅
```
ABLATION TEST (With vs Without Domains)
========================================================
Mean absolute difference:    0.0234
Max absolute difference:     0.1567
Predictions changed >1%:     45.23%

Interpretation:
  ✓  GOOD: Domains have meaningful impact
```

### Poor Domain Usage ❌
```
ABLATION TEST (With vs Without Domains)
========================================================
Mean absolute difference:    0.0003
Max absolute difference:     0.0089
Predictions changed >1%:     2.34%

Interpretation:
  ⚠️  WARNING: Domains have minimal impact (<0.1%)
      Model may not be using domain features!

Recommendations:
  1. Use ImprovedProteinClassifierWithDomains with explicit attention
  2. Increase domain_output_dim (try 512 instead of 256)
  3. Add separate loss term for domain predictions
```

---

## 🛠️ If Domains Aren't Being Used

The diagnostics will tell you automatically, but here are the fixes:

1. **Switch to improved model** (easiest)
   ```python
   model = ImprovedProteinClassifierWithDomains(...)
   ```

2. **Increase domain feature dimension**
   ```python
   CONFIG['domain_output_dim'] = 512  # Was 256
   ```

3. **Use separate learning rates**
   ```python
   optimizer = optim.AdamW([
       {'params': model.domain_encoder.parameters(), 'lr': 1e-3},  # Higher
       {'params': other_params, 'lr': 5e-4}  # Normal
   ])
   ```

4. **Add domain-specific regularization** (see guide for code)

---

## 📖 Documentation

- **Quick start**: This file
- **Detailed integration**: `DOMAIN_FEATURE_INTEGRATION_GUIDE.md`
- **Exact code changes**: `CHANGES_TO_MAIN_SCRIPT.md`
- **Standalone test**: `python test_domain_usage.py`

---

## ⚡ TL;DR

1. **Problem**: You don't know if domains are being used
2. **Solution**: Run `ablation_test()` to check
3. **If not used**: Switch to `ImprovedProteinClassifierWithDomains`
4. **Monitor**: Use `FeatureUsageMonitor` during training
5. **Result**: Know exactly how much domains contribute

**Start with just one line:**
```python
from domain_diagnostics import ablation_test
ablation_test(model, val_loader, device)
```

This will immediately tell you if you have a problem. If you do, the output will tell you exactly how to fix it.

---

## ❓ FAQ

**Q: Will this slow down training?**
A: The diagnostics run after training, so no impact on training speed. Monitoring adds < 0.1% overhead.

**Q: Do I need to retrain from scratch?**
A: To use the improved model, yes. But run diagnostics on your current model first to see if you need to.

**Q: What if I have a pre-trained model?**
A: Run `ablation_test(model, val_loader, device)` on it to check if domains were used.

**Q: Can I use this with other features (not just domains)?**
A: Yes! The same approach works for any auxiliary features (taxonomy, structure, etc.)

**Q: What's a "good" domain importance score?**
A: Mean difference > 0.01 is good, > 0.03 is excellent. Lower means domains aren't helping much.

---

## 🎯 Bottom Line

**Your current code concatenates features, but that doesn't guarantee they're used!**

These tools:
1. ✅ Show you if domains are actually being used
2. ✅ Quantify how much they contribute
3. ✅ Provide an improved model if they're not being used
4. ✅ Let you monitor feature usage during training

**Run the diagnostics, get immediate feedback, fix if needed.**
