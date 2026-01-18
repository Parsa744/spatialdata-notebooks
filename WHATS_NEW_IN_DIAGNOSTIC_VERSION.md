# What's New in train_v10_domains_with_diagnostics.py

## Summary

This is your original training script **with domain diagnostics integrated**. It will automatically check if domain features are being used during training and give you clear feedback.

## What Changed

### 1. Import Domain Diagnostics (Lines 24-31)

```python
# NEW: Import diagnostic tools to verify domain feature usage
try:
    from domain_diagnostics import ablation_test, compute_feature_importance
    DIAGNOSTICS_AVAILABLE = True
    print("✓ Domain diagnostics loaded successfully")
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    print("⚠ Domain diagnostics not available (domain_diagnostics.py not found)")
```

**What this does**: Safely imports diagnostic tools. If `domain_diagnostics.py` is not found, the script still works normally.

---

### 2. Run Diagnostics After Training Each Model (Lines ~850-890 in train_all())

**After each model is trained**, this code block is added:

```python
# ===== NEW: RUN DOMAIN DIAGNOSTICS =====
domain_impact = None
domain_used = False

if DIAGNOSTICS_AVAILABLE and USE_DOMAINS:
    print(f"\n{'='*80}")
    print(f"🔍 DOMAIN FEATURE DIAGNOSTICS FOR {aspect}")
    print(f"{'='*80}")

    try:
        # Run ablation test to check if domains are being used
        ablation_result = ablation_test(model, val_loader, device)
        domain_impact = ablation_result['mean_diff']
        domain_used = domain_impact > 0.01

        # Print summary
        print(f"\n📊 Domain Impact Summary:")
        print(f"   Mean prediction change: {domain_impact:.4f} ({domain_impact*100:.2f}%)")
        print(f"   Predictions changed >1%: {ablation_result['percent_changed']:.1f}%")
        print(f"   Domain features {'ARE' if domain_used else 'ARE NOT'} being used effectively")

        if not domain_used:
            print(f"\n⚠️  WARNING: Domains have minimal impact!")
            print(f"   Consider:")
            print(f"   1. Using ImprovedProteinClassifierWithDomains")
            print(f"   2. Increasing domain_output_dim to 512")
        else:
            print(f"\n✅ Domains are contributing meaningfully!")

    except Exception as e:
        print(f"⚠️  Diagnostic test failed: {e}")
# ===== END DIAGNOSTICS =====
```

**What this does**:
- Runs ablation test to compare predictions WITH vs WITHOUT domains
- Calculates domain impact score
- Prints clear warnings if domains aren't being used
- Gives recommendations for fixes

---

### 3. Store Diagnostic Results (Lines ~895)

```python
results.append({
    'aspect': aspect,
    'name': name,
    'fmax': fmax,
    'threshold': threshold,
    'classes': num_classes,
    'domain_impact': domain_impact,      # NEW
    'domain_used': domain_used            # NEW
})
```

**What this does**: Saves domain impact metrics for each aspect.

---

### 4. Enhanced Summary Report (Lines ~905-925)

```python
print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
for r in results:
    domain_status = ""
    if r['domain_impact'] is not None:
        impact_pct = r['domain_impact'] * 100
        status_icon = "✅" if r['domain_used'] else "⚠️ "
        domain_status = f" | Domains: {status_icon} {impact_pct:.2f}%"  # NEW

    print(f"{r['aspect']:<4} {r['name']:<25} Fmax={r['fmax']:.4f} "
          f"t={r['threshold']:.2f} classes={r['classes']}{domain_status}")

# NEW: Domain usage summary
if any(r['domain_impact'] is not None for r in results):
    print(f"\n{'='*80}\nDOMAIN FEATURE USAGE SUMMARY\n{'='*80}")
    all_used = all(r['domain_used'] for r in results if r['domain_impact'] is not None)

    if all_used:
        print("✅ All aspects are using domain features effectively!")
    else:
        print("⚠️  WARNING: Domain features are NOT being used!")
        print("   See README_DOMAIN_SOLUTION.md for fixes")
```

**What this does**:
- Shows domain impact percentage for each aspect in the summary
- Provides overall assessment of domain usage
- Gives clear next steps if domains aren't being used

---

## Example Output

### When Domains ARE Being Used ✅

```
================================================================================
🔍 DOMAIN FEATURE DIAGNOSTICS FOR F
================================================================================

ABLATION TEST (With vs Without Domains)
============================================================
Mean absolute difference:    0.0234
Max absolute difference:     0.1567
Predictions changed >1%:     45.23%

Interpretation:
  ✓  GOOD: Domains have meaningful impact
============================================================

📊 Domain Impact Summary:
   Mean prediction change: 0.0234 (2.34%)
   Predictions changed >1%: 45.2%
   Domain features ARE being used effectively

✅ Domains are contributing meaningfully!

================================================================================
SUMMARY
================================================================================
F    Molecular Function       Fmax=0.4567 t=0.18 classes=4523 | Domains: ✅ 2.34%
C    Cellular Component       Fmax=0.5234 t=0.22 classes=892  | Domains: ✅ 1.89%
P    Biological Process       Fmax=0.3891 t=0.15 classes=9876 | Domains: ✅ 2.01%

AVG Fmax: 0.4564

================================================================================
DOMAIN FEATURE USAGE SUMMARY
================================================================================
✅ All aspects are using domain features effectively!
```

---

### When Domains ARE NOT Being Used ⚠️

```
================================================================================
🔍 DOMAIN FEATURE DIAGNOSTICS FOR F
================================================================================

ABLATION TEST (With vs Without Domains)
============================================================
Mean absolute difference:    0.0003
Max absolute difference:     0.0089
Predictions changed >1%:     2.34%

Interpretation:
  ⚠️  WARNING: Domains have minimal impact (<0.1%)
      Model may not be using domain features!
============================================================

📊 Domain Impact Summary:
   Mean prediction change: 0.0003 (0.03%)
   Predictions changed >1%: 2.3%
   Domain features ARE NOT being used effectively

⚠️  WARNING: Domains have minimal impact!
   Consider:
   1. Using ImprovedProteinClassifierWithDomains
   2. Increasing domain_output_dim to 512
   3. See CHANGES_TO_MAIN_SCRIPT.md for fixes

================================================================================
SUMMARY
================================================================================
F    Molecular Function       Fmax=0.4521 t=0.18 classes=4523 | Domains: ⚠️  0.03%
C    Cellular Component       Fmax=0.5187 t=0.22 classes=892  | Domains: ⚠️  0.01%
P    Biological Process       Fmax=0.3845 t=0.15 classes=9876 | Domains: ⚠️  0.02%

AVG Fmax: 0.4518

================================================================================
DOMAIN FEATURE USAGE SUMMARY
================================================================================
⚠️  WARNING: Domain features are NOT being used in any aspect!
   See README_DOMAIN_SOLUTION.md for fixes
```

---

## How to Use

1. **Make sure `domain_diagnostics.py` is in the same directory**

2. **Run the script normally:**
   ```bash
   python train_v10_domains_with_diagnostics.py
   ```

3. **Check the output after each aspect is trained**
   - Look for the `🔍 DOMAIN FEATURE DIAGNOSTICS` section
   - Check if domains are being used (✅) or not (⚠️)

4. **Read the final summary**
   - At the end, you'll see overall domain usage across all aspects
   - If domains aren't being used, you'll get clear instructions

---

## If Domains Aren't Being Used

The script will tell you to:

1. **Use the improved model**: `ImprovedProteinClassifierWithDomains` from `domain_diagnostics.py`
2. **Increase domain dimensions**: Set `domain_output_dim: 512` in CONFIG
3. **Read the fix guide**: See `CHANGES_TO_MAIN_SCRIPT.md` for detailed fixes

---

## Comparison with Original Script

| Feature | Original Script | Diagnostic Version |
|---------|----------------|-------------------|
| Trains models | ✅ | ✅ |
| Uses domain features | ✅ | ✅ |
| Verifies domain usage | ❌ | ✅ |
| Shows domain impact | ❌ | ✅ |
| Gives improvement suggestions | ❌ | ✅ |
| Works without diagnostics | N/A | ✅ (graceful fallback) |

---

## Files You Need

1. `train_v10_domains_with_diagnostics.py` (this script)
2. `domain_diagnostics.py` (diagnostic tools)
3. All your data files (same as before)

---

## No Changes Needed to Your Data or Setup

Everything else stays exactly the same:
- Same data paths
- Same model architecture (unless you choose to upgrade)
- Same hyperparameters
- Same training process

The only difference: **You'll know if domains are actually helping!**

---

## Bottom Line

Run this instead of your original script, and you'll get **automatic verification** that domain features are being used. No guesswork, no uncertainty—just clear feedback and actionable recommendations.

**Before**: "I think domains are being used... maybe?"
**After**: "Domain impact: 2.34% ✅ Domains are working!"
