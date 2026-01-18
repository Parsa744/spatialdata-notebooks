#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Domain Feature Usage Test
================================
Run this script to immediately check if your model is using domain features.

Usage:
    python test_domain_usage.py

This will load your trained model and run diagnostics to show:
1. Whether domains are being used
2. How much impact they have
3. Recommendations for improvement
"""

import sys
import os

# Try to import the diagnostics
try:
    from domain_diagnostics import run_all_diagnostics, ablation_test, compute_feature_importance
    print("✓ Diagnostic tools loaded successfully\n")
except ImportError:
    print("ERROR: Could not import domain_diagnostics.py")
    print("Make sure domain_diagnostics.py is in the same directory")
    sys.exit(1)

# Try to import your main training script components
# Adjust these imports based on your script name
try:
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    print("✓ PyTorch loaded\n")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


def quick_test_with_sample_data():
    """
    If you don't have a trained model yet, this creates dummy data to test the diagnostics.
    """
    print("="*80)
    print("QUICK TEST WITH SAMPLE DATA")
    print("="*80)
    print("\nCreating sample model and data...\n")

    from domain_diagnostics import ImprovedProteinClassifierWithDomains
    import torch.nn as nn
    from torch.utils.data import TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample parameters
    esm_dim = 1280
    num_domains = 500
    num_classes = 100
    batch_size = 32
    num_samples = 256

    # Create sample model
    model = ImprovedProteinClassifierWithDomains(
        esm_dim=esm_dim,
        num_classes=num_classes,
        hidden_dims=[1024, 512, 256],
        dropout=0.2,
        num_domains=num_domains,
        domain_output_dim=256,
        use_domains=True
    ).to(device)

    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create sample data
    X_esm = torch.randn(num_samples, esm_dim)
    X_domain = torch.zeros(num_samples, num_domains)

    # Add some domains to half the samples
    for i in range(num_samples // 2):
        num_domains_sample = np.random.randint(1, 10)
        domain_indices = np.random.choice(num_domains, num_domains_sample, replace=False)
        X_domain[i, domain_indices] = np.random.uniform(0.3, 1.0, num_domains_sample)

    y = torch.zeros(num_samples, num_classes)
    for i in range(num_samples):
        num_labels = np.random.randint(1, 5)
        label_indices = np.random.choice(num_classes, num_labels, replace=False)
        y[i, label_indices] = 1.0

    # Create dataloader
    dataset = TensorDataset(X_esm, X_domain, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Sample data created: {num_samples} samples")
    print(f"  - {(X_domain.sum(dim=1) > 0).sum().item()} samples with domains")
    print(f"  - {X_domain.sum().item():.0f} total domain annotations\n")

    # Run diagnostics
    print("Running diagnostics...\n")
    results = run_all_diagnostics(model, dataloader, device)

    return results


def test_your_trained_model(model, val_loader, device):
    """
    Test your actual trained model.

    Args:
        model: Your trained ProteinClassifierWithDomains model
        val_loader: Your validation DataLoader
        device: torch.device
    """
    print("="*80)
    print("TESTING YOUR TRAINED MODEL")
    print("="*80)
    print()

    results = run_all_diagnostics(model, val_loader, device)

    return results


def main():
    """
    Main entry point for quick testing.
    """
    print("\n" + "="*80)
    print("DOMAIN FEATURE USAGE DIAGNOSTIC TOOL")
    print("="*80)
    print()
    print("This tool will check if your model is using domain features effectively.")
    print()

    # Check if user wants to test with sample data or their own model
    print("Options:")
    print("  1. Quick test with sample data (no trained model needed)")
    print("  2. Test your trained model (requires model and data)")
    print()

    choice = input("Enter choice (1 or 2, or Enter for option 1): ").strip()

    if choice == "2":
        print("\nTo test your trained model, you need to:")
        print("  1. Import your model: aspect_models[aspect]")
        print("  2. Import your validation loader: val_loader")
        print("  3. Call: test_your_trained_model(model, val_loader, device)")
        print()
        print("Example code:")
        print("-" * 60)
        print("""
# In your main training script, after training:
from test_domain_usage import test_your_trained_model

for aspect in ['F', 'C', 'P']:
    print(f"\\nTesting {aspect} model...")
    test_your_trained_model(
        aspect_models[aspect],
        val_loader,  # Your validation DataLoader for this aspect
        device
    )
        """)
        print("-" * 60)
    else:
        # Run quick test
        results = quick_test_with_sample_data()

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("""
1. To test your actual model, add this to your training script:

   from test_domain_usage import test_your_trained_model

   # After training each aspect:
   test_your_trained_model(model, val_loader, device)

2. If domains aren't being used, try:

   from domain_diagnostics import ImprovedProteinClassifierWithDomains

   # Replace your model with the improved version
   model = ImprovedProteinClassifierWithDomains(...)

3. Monitor feature usage during training:

   from domain_diagnostics import FeatureUsageMonitor

   monitor = FeatureUsageMonitor()
   # In training loop:
   monitor.log(epoch, model)
   # After training:
   monitor.plot(aspect)

See DOMAIN_FEATURE_INTEGRATION_GUIDE.md for detailed instructions.
        """)


if __name__ == "__main__":
    main()
