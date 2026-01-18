# 🚀 Quick Start: Verify Domain Features Are Being Used

## Your Problem

You have ESM embeddings + InterPro/Signature domain features, but you're not sure if the model is actually using the domain information or just relying on ESM embeddings.

## The Solution (2 options)

### ⚡ Option 1: Use the Updated Script (Easiest)

**Just run this instead of your original script:**

```bash
python train_v10_domains_with_diagnostics.py
```

**What happens:**
- Trains exactly like your original script
- Automatically tests domain usage after each model
- Shows clear ✅/⚠️ indicators
- Gives recommendations if domains aren't being used

**You'll see output like:**

```
🔍 DOMAIN FEATURE DIAGNOSTICS FOR F
============================================================
Mean absolute difference:    0.0234

Domain features ARE being used effectively ✅

SUMMARY
F    Molecular Function    Fmax=0.4567 | Domains: ✅ 2.34%
C    Cellular Component    Fmax=0.5234 | Domains: ✅ 1.89%
P    Biological Process    Fmax=0.3891 | Domains: ✅ 2.01%
```

**If you see ⚠️ instead:** Follow the on-screen recommendations to fix it.

---

### 🔧 Option 2: Manual Integration (If you want to modify your existing script)

See `CHANGES_TO_MAIN_SCRIPT.md` for line-by-line instructions to add diagnostics to your current script.

---

## Files You Need

| File | Purpose | Priority |
|------|---------|----------|
| `train_v10_domains_with_diagnostics.py` | Updated training script with diagnostics | ⭐⭐⭐ Use this |
| `domain_diagnostics.py` | Diagnostic tools (imported by above) | ⭐⭐⭐ Required |
| `README_DOMAIN_SOLUTION.md` | Complete solution guide | ⭐⭐ Read if you want details |
| `WHATS_NEW_IN_DIAGNOSTIC_VERSION.md` | What changed in the updated script | ⭐ Reference |
| `CHANGES_TO_MAIN_SCRIPT.md` | Manual integration guide | ⭐ If customizing |

---

## What You'll Learn

After running the diagnostic script, you'll know:

1. **Are domains being used?** (Yes/No with confidence)
2. **How much impact?** (Exact percentage)
3. **What to do if not?** (Clear recommendations)

---

## Example: What to Expect

### ✅ Good Result (Domains Working)

```
Domain Impact Summary:
   Mean prediction change: 0.0234 (2.34%)
   Domain features ARE being used effectively

✅ Domains are contributing meaningfully!
```

**Meaning**: Your model is using domain features. The domains change predictions by ~2.34% on average. **You're good!**

---

### ⚠️ Problem Detected (Domains Ignored)

```
Domain Impact Summary:
   Mean prediction change: 0.0003 (0.03%)
   Domain features ARE NOT being used effectively

⚠️  WARNING: Domains have minimal impact!
   Consider:
   1. Using ImprovedProteinClassifierWithDomains
   2. Increasing domain_output_dim to 512
```

**Meaning**: Domains are concatenated but the model ignores them. Impact is < 0.1%. **Apply the recommended fixes.**

---

## If Domains Aren't Being Used

The script will tell you what to do, but here's the quick version:

1. **Use the improved model** with explicit attention:
   ```python
   from domain_diagnostics import ImprovedProteinClassifierWithDomains
   model = ImprovedProteinClassifierWithDomains(...)  # Replace your current model
   ```

2. **Increase domain feature dimensions:**
   ```python
   CONFIG['domain_output_dim'] = 512  # Was 256
   ```

3. **See the detailed fix guide:** `README_DOMAIN_SOLUTION.md`

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'domain_diagnostics'"

**Fix**: Make sure `domain_diagnostics.py` is in the same directory as your training script.

```bash
ls -l domain_diagnostics.py  # Should exist
```

### "Diagnostics not available"

**Result**: Script runs normally but skips diagnostics.

**Fix**: Download `domain_diagnostics.py` to the same directory.

### Domains show 0.00% impact

**This is the problem we're solving!** Follow the recommendations shown in the output.

---

## The Bottom Line

### Before (Your Current Situation)
```
Training completes...
Model uses ESM + Domains (you hope)
??? Are domains actually helping? Unknown!
```

### After (With Diagnostics)
```
Training completes...
🔍 Running diagnostics...
✅ Domains: 2.34% impact - Working!
or
⚠️  Domains: 0.03% impact - Not working! Fix: [specific steps]
```

**No more guessing. Know for certain.**

---

## Summary Commands

```bash
# 1. Make sure you have the files
ls domain_diagnostics.py train_v10_domains_with_diagnostics.py

# 2. Run the diagnostic version
python train_v10_domains_with_diagnostics.py

# 3. Check the output for ✅ or ⚠️ indicators

# 4. If ⚠️, apply recommended fixes and rerun
```

---

## Need More Details?

- **Quick overview**: This file (you're reading it)
- **What changed**: `WHATS_NEW_IN_DIAGNOSTIC_VERSION.md`
- **Complete guide**: `README_DOMAIN_SOLUTION.md`
- **Manual integration**: `CHANGES_TO_MAIN_SCRIPT.md`
- **Technical details**: `DOMAIN_FEATURE_INTEGRATION_GUIDE.md`

---

## Contact / Issues

If something doesn't work as expected:

1. Check that `domain_diagnostics.py` is in the same directory
2. Review the error message carefully
3. See `README_DOMAIN_SOLUTION.md` for troubleshooting
4. Check that all your data paths are correct (same as original script)

---

**Start here**: Just run `train_v10_domains_with_diagnostics.py` and read the output. It will tell you everything you need to know!
