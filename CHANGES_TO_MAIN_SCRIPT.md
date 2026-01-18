# Exact Changes to Make to Your Main Script

## Change 1: Add Imports (at the top of your script)

```python
# ADD THESE LINES after your existing imports:
from domain_diagnostics import (
    run_all_diagnostics,
    ImprovedProteinClassifierWithDomains,
    FeatureUsageMonitor,
    compute_feature_importance,
    ablation_test
)
```

---

## Change 2: Modify train_model function (OPTION A: Quick diagnostic)

### ADD THIS CODE at the end of train_model(), before the return statement:

```python
def train_model(aspect, model, train_loader, valid_loader, ia_weights_tensor, config, class_weights=None):
    # ... all your existing training code ...

    # [EXISTING CODE ENDS HERE - typically after plotting]

    # ===== ADD THIS BLOCK =====
    print(f"\n{'='*80}")
    print(f"DOMAIN FEATURE DIAGNOSTICS FOR {aspect}")
    print(f"{'='*80}")

    # Quick ablation test
    ablation_result = ablation_test(model, valid_loader, device)

    # Store in history for logging
    if 'domain_impact' not in history:
        history['domain_impact'] = ablation_result['mean_diff']
        history['domain_used'] = ablation_result['mean_diff'] > 0.01

    # ===== END NEW BLOCK =====

    return model, best_fmax, best_t
```

---

## Change 3: Replace Model (OPTION B: Full solution with attention)

### FIND THIS CODE in train_all():

```python
# OLD CODE - FIND THIS:
model = ProteinClassifierWithDomains(
    EMBED_DIM, num_classes, config['hidden_dims'], config['dropout'],
    num_domains=NUM_DOMAINS, domain_output_dim=CONFIG['domain_output_dim'],
    use_domains=USE_DOMAINS
)
```

### REPLACE WITH:

```python
# NEW CODE - REPLACE WITH THIS:
model = ImprovedProteinClassifierWithDomains(  # <-- Changed class name
    esm_dim=EMBED_DIM,  # <-- Added parameter name
    num_classes=num_classes,
    hidden_dims=config['hidden_dims'],
    dropout=config['dropout'],
    num_domains=NUM_DOMAINS,
    domain_output_dim=CONFIG['domain_output_dim'],
    use_domains=USE_DOMAINS
)
```

---

## Change 4: Add Feature Usage Monitoring

### FIND THIS CODE in train_model():

```python
# FIND THE START of train_model function:
def train_model(aspect, model, train_loader, valid_loader, ia_weights_tensor, config, class_weights=None):
    model = model.to(device)
    # ... more setup code ...
```

### ADD THIS RIGHT AFTER model.to(device):

```python
def train_model(aspect, model, train_loader, valid_loader, ia_weights_tensor, config, class_weights=None):
    model = model.to(device)

    # ===== ADD THIS BLOCK =====
    feature_monitor = FeatureUsageMonitor()
    # ===== END NEW BLOCK =====

    ia_weights_tensor = ia_weights_tensor.to(device)
    # ... rest of function ...
```

### THEN FIND THIS CODE in the training loop (inside the epoch loop):

```python
    # After validation, you have code like:
    fmax_w, t_w, prec_w, rec_w = compute_fmax_adaptive(all_preds, all_labels, ia_weights_tensor, use_ia=True)
    # ... more validation code ...
    print(f"[{aspect}] Ep {epoch:2d} | LR {lr:.1e} | ...")
```

### ADD THIS RIGHT AFTER the print statement:

```python
    print(f"[{aspect}] Ep {epoch:2d} | LR {lr:.1e} | ...")

    # ===== ADD THIS BLOCK =====
    # Log feature usage
    feature_monitor.log(epoch, model)

    # Print feature weights every 5 epochs
    if epoch % 5 == 0:
        weights = model.get_feature_weights()
        if weights:
            print(f"    💡 Feature weights - ESM: {weights['esm_weight']:.3f}, "
                  f"Domain: {weights['domain_weight']:.3f}")
    # ===== END NEW BLOCK =====
```

### FINALLY, ADD THIS at the very end of train_model(), after plotting but before return:

```python
    plt.tight_layout()
    plt.savefig(f'{aspect}_training_v10.png', dpi=150)
    plt.show()

    # ===== ADD THIS BLOCK =====
    # Plot feature usage over training
    feature_monitor.plot(aspect)
    # ===== END NEW BLOCK =====

    return model, best_fmax, best_t
```

---

## Complete Modified train_model() Function Skeleton

Here's what the modified function looks like:

```python
def train_model(aspect, model, train_loader, valid_loader, ia_weights_tensor, config, class_weights=None):
    model = model.to(device)

    # NEW: Add feature monitoring
    feature_monitor = FeatureUsageMonitor()

    ia_weights_tensor = ia_weights_tensor.to(device)
    # ... existing setup code ...

    for epoch in range(1, config['epochs'] + 1):
        # ... existing training code ...

        # ... existing validation code ...

        # ... existing metric computation and printing ...
        print(f"[{aspect}] Ep {epoch:2d} | LR {lr:.1e} | ...")

        # NEW: Log and print feature usage
        feature_monitor.log(epoch, model)
        if epoch % 5 == 0:
            weights = model.get_feature_weights()
            if weights:
                print(f"    💡 Feature weights - ESM: {weights['esm_weight']:.3f}, "
                      f"Domain: {weights['domain_weight']:.3f}")

        # ... existing early stopping code ...

    # ... existing plotting code ...

    # NEW: Plot feature usage
    feature_monitor.plot(aspect)

    # NEW: Run diagnostics
    print(f"\n{'='*80}")
    print(f"DOMAIN FEATURE DIAGNOSTICS FOR {aspect}")
    print(f"{'='*80}")
    ablation_result = ablation_test(model, valid_loader, device)

    return model, best_fmax, best_t
```

---

## Quick Start: Minimal Changes

If you want the MINIMUM changes to just check if domains are being used:

### 1. Add one import:
```python
from domain_diagnostics import ablation_test
```

### 2. Add one function call in train_all(), after each model is trained:
```python
for aspect in ASPECTS:
    # ... existing training code ...
    model, fmax, threshold = train_model(...)

    # ADD THIS LINE:
    ablation_test(model, val_loader, device)

    # ... rest of code ...
```

That's it! This will tell you immediately if domains are being used.

---

## Expected Console Output

### With Diagnostics Enabled:

```
[P] Ep  1 | LR 5.0e-04 | Train 0.1234 | Val 0.1156 | Fmax(IA) 0.3421 (t=0.12) | ...
[P] Ep  5 | LR 4.8e-04 | Train 0.0987 | Val 0.0945 | Fmax(IA) 0.4123 (t=0.15) | ...
    💡 Feature weights - ESM: 0.687, Domain: 0.313    <-- YOU'LL SEE THIS
[P] Ep 10 | LR 4.5e-04 | Train 0.0854 | Val 0.0823 | Fmax(IA) 0.4567 (t=0.18) | ...
    💡 Feature weights - ESM: 0.702, Domain: 0.298    <-- AND THIS
...

================================================================================
DOMAIN FEATURE DIAGNOSTICS FOR P
================================================================================

ABLATION TEST (With vs Without Domains)
============================================================
Mean absolute difference:    0.0234                       <-- IMPACT MEASURE
Max absolute difference:     0.1567
Predictions changed >1%:     45.23%

Interpretation:
  ✓  GOOD: Domains have meaningful impact                <-- ASSESSMENT
============================================================
```

---

## Troubleshooting

### If you get: `NameError: name 'FeatureUsageMonitor' is not defined`
→ Make sure you added the import at the top

### If you get: `AttributeError: 'ProteinClassifierWithDomains' object has no attribute 'get_feature_weights'`
→ You need to use `ImprovedProteinClassifierWithDomains` instead (see Change 3)

### If diagnostics show domains aren't being used:
→ See DOMAIN_FEATURE_INTEGRATION_GUIDE.md for fixes

---

## Summary

| Change | Difficulty | Benefit |
|--------|-----------|---------|
| Add ablation_test() call | Easy (1 line) | Quick check if domains are used |
| Replace with ImprovedModel | Medium (modify 1 class) | Explicit attention, better domain usage |
| Add FeatureUsageMonitor | Medium (3 code blocks) | See feature weights during training |
| Full integration | Medium (all changes above) | Complete visibility and optimization |

**Recommended approach:**
1. Start with just `ablation_test()` to diagnose the problem
2. If domains aren't being used, switch to `ImprovedProteinClassifierWithDomains`
3. Add `FeatureUsageMonitor` to track improvements
4. Retrain and verify domains are now being used effectively
