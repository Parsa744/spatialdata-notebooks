#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Domain Feature Diagnostics and Improvements
============================================
Solutions to verify and improve domain feature usage in the model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# ============================================================================
# SOLUTION 1: Feature Importance via Gradient Attribution
# ============================================================================
def compute_feature_importance(model, dataloader, device, num_samples=100):
    """
    Compute gradient-based feature importance for ESM vs Domain features.
    Shows which features the model relies on most.
    """
    model.eval()
    esm_grads = []
    domain_grads = []

    samples_processed = 0

    for X_esm, X_domain, y in dataloader:
        if samples_processed >= num_samples:
            break

        X_esm = X_esm.to(device).requires_grad_(True)
        X_domain = X_domain.to(device).requires_grad_(True)

        # Forward pass
        outputs = model(X_esm, domain_multihot=X_domain)

        # Backward on the predictions
        loss = outputs.sum()
        loss.backward()

        # Collect gradients (absolute values indicate importance)
        esm_grads.append(X_esm.grad.abs().mean(dim=0).cpu().numpy())
        domain_grads.append(X_domain.grad.abs().mean(dim=0).cpu().numpy())

        samples_processed += X_esm.shape[0]

        # Clear gradients
        X_esm.grad = None
        X_domain.grad = None

    # Average importance scores
    esm_importance = np.mean([g.mean() for g in esm_grads])
    domain_importance = np.mean([g.mean() for g in domain_grads])

    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS (Gradient-based)")
    print(f"{'='*60}")
    print(f"ESM Embedding Importance:    {esm_importance:.6f}")
    print(f"Domain Feature Importance:   {domain_importance:.6f}")
    print(f"Domain/ESM Ratio:            {domain_importance/esm_importance:.4f}")
    print(f"{'='*60}\n")

    return {
        'esm_importance': esm_importance,
        'domain_importance': domain_importance,
        'ratio': domain_importance / esm_importance
    }


# ============================================================================
# SOLUTION 2: Ablation Test - Compare with/without domains
# ============================================================================
def ablation_test(model, dataloader, device):
    """
    Compare model predictions with and without domain features.
    If predictions are similar, domains aren't being used.
    """
    model.eval()

    all_preds_with = []
    all_preds_without = []

    with torch.no_grad():
        for X_esm, X_domain, y in dataloader:
            X_esm = X_esm.to(device)
            X_domain = X_domain.to(device)

            # Predictions WITH domains
            preds_with = torch.sigmoid(model(X_esm, domain_multihot=X_domain))

            # Predictions WITHOUT domains (zero out domain features)
            zero_domains = torch.zeros_like(X_domain)
            preds_without = torch.sigmoid(model(X_esm, domain_multihot=zero_domains))

            all_preds_with.append(preds_with.cpu())
            all_preds_without.append(preds_without.cpu())

    all_preds_with = torch.cat(all_preds_with, dim=0)
    all_preds_without = torch.cat(all_preds_without, dim=0)

    # Calculate differences
    abs_diff = (all_preds_with - all_preds_without).abs()
    mean_diff = abs_diff.mean().item()
    max_diff = abs_diff.max().item()
    percent_changed = (abs_diff > 0.01).float().mean().item() * 100

    print(f"\n{'='*60}")
    print(f"ABLATION TEST (With vs Without Domains)")
    print(f"{'='*60}")
    print(f"Mean absolute difference:    {mean_diff:.6f}")
    print(f"Max absolute difference:     {max_diff:.6f}")
    print(f"Predictions changed >1%:     {percent_changed:.2f}%")
    print(f"\nInterpretation:")
    if mean_diff < 0.001:
        print("  ⚠️  WARNING: Domains have minimal impact (<0.1%)")
        print("      Model may not be using domain features!")
    elif mean_diff < 0.01:
        print("  ⚠️  WEAK: Domains have small impact (<1%)")
        print("      Consider increasing domain feature weight")
    else:
        print("  ✓  GOOD: Domains have meaningful impact")
    print(f"{'='*60}\n")

    return {
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'percent_changed': percent_changed
    }


# ============================================================================
# SOLUTION 3: Improved Model with Explicit Domain Attention
# ============================================================================
class FeatureFusionWithAttention(nn.Module):
    """
    Explicitly shows how much the model attends to ESM vs Domain features.
    """
    def __init__(self, esm_dim, domain_dim, output_dim):
        super().__init__()

        # Separate processing paths
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        self.domain_proj = nn.Sequential(
            nn.Linear(domain_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        # Cross-attention between features
        self.esm_attn = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )

        self.domain_attn = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )

        # Store attention weights for visualization
        self.last_esm_weight = None
        self.last_domain_weight = None

    def forward(self, esm_feat, domain_feat):
        # Project features
        esm = self.esm_proj(esm_feat)
        domain = self.domain_proj(domain_feat)

        # Compute attention weights
        esm_weight = self.esm_attn(esm)
        domain_weight = self.domain_attn(domain)

        # Normalize weights
        total_weight = esm_weight + domain_weight + 1e-8
        esm_weight = esm_weight / total_weight
        domain_weight = domain_weight / total_weight

        # Store for logging
        self.last_esm_weight = esm_weight.detach().mean().item()
        self.last_domain_weight = domain_weight.detach().mean().item()

        # Weighted fusion
        fused = esm_weight * esm + domain_weight * domain

        return fused


class ImprovedProteinClassifierWithDomains(nn.Module):
    """
    Improved version with explicit feature fusion and attention tracking.
    """
    def __init__(self, esm_dim, num_classes, hidden_dims, dropout=0.2,
                 num_domains=0, domain_output_dim=256, use_domains=True):
        super().__init__()
        self.use_domains = use_domains and num_domains > 0

        if self.use_domains:
            # Domain encoder
            self.domain_encoder = nn.Sequential(
                nn.Linear(num_domains, domain_output_dim * 2),
                nn.LayerNorm(domain_output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(domain_output_dim * 2, domain_output_dim),
                nn.LayerNorm(domain_output_dim),
            )

            # Feature fusion with attention
            self.feature_fusion = FeatureFusionWithAttention(
                esm_dim, domain_output_dim, hidden_dims[0]
            )

            print(f"    Using EXPLICIT feature fusion with attention")
            print(f"    ESM: {esm_dim} -> {hidden_dims[0]}")
            print(f"    Domains: {num_domains} -> {domain_output_dim} -> {hidden_dims[0]}")
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(esm_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # Rest of the network (same as before)
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        self.output = nn.Linear(hidden_dims[-1], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, domain_multihot=None):
        if self.use_domains and domain_multihot is not None:
            domain_feat = self.domain_encoder(domain_multihot)
            x = self.feature_fusion(x, domain_feat)
        else:
            x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return self.output(x)

    def get_feature_weights(self):
        """Get the current attention weights for ESM vs Domain features."""
        if self.use_domains and hasattr(self.feature_fusion, 'last_esm_weight'):
            return {
                'esm_weight': self.feature_fusion.last_esm_weight,
                'domain_weight': self.feature_fusion.last_domain_weight
            }
        return None


# ============================================================================
# SOLUTION 4: Training with Feature Usage Monitoring
# ============================================================================
class FeatureUsageMonitor:
    """
    Tracks feature importance during training.
    """
    def __init__(self):
        self.history = defaultdict(list)

    def log(self, epoch, model):
        """Log feature weights after each epoch."""
        weights = model.get_feature_weights()
        if weights:
            self.history['epoch'].append(epoch)
            self.history['esm_weight'].append(weights['esm_weight'])
            self.history['domain_weight'].append(weights['domain_weight'])

    def plot(self, aspect=''):
        """Visualize feature usage over training."""
        if not self.history['epoch']:
            print("No feature usage data to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epoch'], self.history['esm_weight'],
                 label='ESM Weight', linewidth=2)
        plt.plot(self.history['epoch'], self.history['domain_weight'],
                 label='Domain Weight', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Attention Weight', fontsize=12)
        plt.title(f'{aspect} Feature Usage During Training', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{aspect}_feature_usage.png', dpi=150)
        plt.show()

        # Print summary
        print(f"\n{'='*60}")
        print(f"FEATURE USAGE SUMMARY - {aspect}")
        print(f"{'='*60}")
        print(f"Average ESM Weight:     {np.mean(self.history['esm_weight']):.4f}")
        print(f"Average Domain Weight:  {np.mean(self.history['domain_weight']):.4f}")
        print(f"Final ESM Weight:       {self.history['esm_weight'][-1]:.4f}")
        print(f"Final Domain Weight:    {self.history['domain_weight'][-1]:.4f}")
        print(f"{'='*60}\n")


# ============================================================================
# SOLUTION 5: Per-Domain Importance Analysis
# ============================================================================
def analyze_domain_contributions(model, dataloader, domain_to_idx, idx_to_domain, device, top_k=20):
    """
    Analyze which specific domains are most important for predictions.
    """
    model.eval()

    domain_contributions = defaultdict(list)

    for X_esm, X_domain, y in dataloader:
        X_esm = X_esm.to(device)
        X_domain = X_domain.to(device).requires_grad_(True)

        # Forward pass
        outputs = model(X_esm, domain_multihot=X_domain)

        # Backward
        loss = outputs.sum()
        loss.backward()

        # Get gradients for each domain
        grads = X_domain.grad.abs().cpu().numpy()

        # For each sample
        for i in range(grads.shape[0]):
            domain_vals = X_domain[i].cpu().numpy()
            domain_grads = grads[i]

            # Find non-zero domains
            for domain_idx in np.where(domain_vals > 0)[0]:
                if domain_idx > 0:  # Skip padding
                    contribution = domain_grads[domain_idx] * domain_vals[domain_idx]
                    domain_contributions[domain_idx].append(contribution)

        X_domain.grad = None

    # Compute average contribution per domain
    domain_importance = {}
    for domain_idx, contributions in domain_contributions.items():
        domain_importance[domain_idx] = np.mean(contributions)

    # Get top domains
    sorted_domains = sorted(domain_importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"TOP {top_k} MOST IMPORTANT DOMAINS")
    print(f"{'='*60}")
    for i, (domain_idx, importance) in enumerate(sorted_domains[:top_k], 1):
        domain_name = idx_to_domain.get(domain_idx, f"Domain_{domain_idx}")
        print(f"{i:2d}. {domain_name:30s} {importance:.6f}")
    print(f"{'='*60}\n")

    return domain_importance


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
def run_all_diagnostics(model, val_loader, device, domain_vocab=None):
    """
    Run all diagnostic tests to verify domain feature usage.
    """
    print("\n" + "="*80)
    print("DOMAIN FEATURE DIAGNOSTICS - COMPREHENSIVE ANALYSIS")
    print("="*80 + "\n")

    # Test 1: Feature Importance
    print("Running Test 1: Gradient-based Feature Importance...")
    importance = compute_feature_importance(model, val_loader, device, num_samples=100)

    # Test 2: Ablation
    print("\nRunning Test 2: Ablation Test...")
    ablation = ablation_test(model, val_loader, device)

    # Test 3: Per-domain analysis (if vocab provided)
    if domain_vocab:
        idx_to_domain = {idx: domain for domain, idx in domain_vocab.items()}
        print("\nRunning Test 3: Per-Domain Importance Analysis...")
        analyze_domain_contributions(model, val_loader, domain_vocab, idx_to_domain, device)

    # Overall assessment
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)

    if ablation['mean_diff'] < 0.001:
        print("🔴 CRITICAL: Domain features are NOT being used effectively!")
        print("\nRecommendations:")
        print("  1. Use ImprovedProteinClassifierWithDomains with explicit attention")
        print("  2. Increase domain_output_dim (try 512 instead of 256)")
        print("  3. Add a separate loss term for domain predictions")
        print("  4. Reduce ESM embedding dimension or add stronger regularization")
    elif ablation['mean_diff'] < 0.01:
        print("🟡 WARNING: Domain features have weak impact")
        print("\nRecommendations:")
        print("  1. Monitor feature weights during training")
        print("  2. Consider feature fusion with attention")
        print("  3. Increase learning rate for domain encoder")
    else:
        print("🟢 GOOD: Domain features are being used!")
        print(f"   Impact level: {ablation['mean_diff']*100:.2f}% average change")

    print("="*80 + "\n")

    return {
        'importance': importance,
        'ablation': ablation
    }


if __name__ == "__main__":
    print(__doc__)
    print("\nImport this module and use:")
    print("  - compute_feature_importance()")
    print("  - ablation_test()")
    print("  - ImprovedProteinClassifierWithDomains")
    print("  - FeatureUsageMonitor")
    print("  - run_all_diagnostics()")
