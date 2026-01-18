#!/usr/bin/env python
"""
Creates visualization showing the difference between current and improved models
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_comparison():
    """Create side-by-side comparison of model architectures"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Colors
    esm_color = '#4A90E2'
    domain_color = '#E67E22'
    concat_color = '#9B59B6'
    output_color = '#27AE60'
    problem_color = '#E74C3C'
    solution_color = '#2ECC71'

    def draw_architecture(ax, is_improved=False):
        """Draw model architecture diagram"""

        # Input features
        esm_box = FancyBboxPatch((0.5, 7), 2, 1, boxstyle="round,pad=0.1",
                                  edgecolor=esm_color, facecolor=esm_color, alpha=0.3, linewidth=2)
        domain_box = FancyBboxPatch((3.5, 7), 2, 1, boxstyle="round,pad=0.1",
                                     edgecolor=domain_color, facecolor=domain_color, alpha=0.3, linewidth=2)
        ax.add_patch(esm_box)
        ax.add_patch(domain_box)
        ax.text(1.5, 7.5, 'ESM\nEmbedding\n(1280)', ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(4.5, 7.5, 'Domain\nFeatures\n(500)', ha='center', va='center', fontsize=11, fontweight='bold')

        if not is_improved:
            # Current model: simple concatenation
            concat_box = FancyBboxPatch((1.5, 5), 2, 1, boxstyle="round,pad=0.1",
                                        edgecolor=concat_color, facecolor=concat_color, alpha=0.3, linewidth=2)
            ax.add_patch(concat_box)
            ax.text(2.5, 5.5, 'Concatenate', ha='center', va='center', fontsize=11, fontweight='bold')

            # Arrows from inputs to concat
            arrow1 = FancyArrowPatch((1.5, 7), (2.2, 6), arrowstyle='->', mutation_scale=20, linewidth=2, color='gray')
            arrow2 = FancyArrowPatch((4.5, 7), (2.8, 6), arrowstyle='->', mutation_scale=20, linewidth=2, color='gray')
            ax.add_patch(arrow1)
            ax.add_patch(arrow2)

            # Hidden layers
            hidden_box = FancyBboxPatch((1.5, 3.5), 2, 1, boxstyle="round,pad=0.1",
                                        edgecolor='gray', facecolor='lightgray', alpha=0.5, linewidth=2)
            ax.add_patch(hidden_box)
            ax.text(2.5, 4, 'Hidden\nLayers', ha='center', va='center', fontsize=11, fontweight='bold')

            # Arrow from concat to hidden
            arrow3 = FancyArrowPatch((2.5, 5), (2.5, 4.5), arrowstyle='->', mutation_scale=20, linewidth=2, color='gray')
            ax.add_patch(arrow3)

            # Output
            output_box = FancyBboxPatch((1.5, 2), 2, 1, boxstyle="round,pad=0.1",
                                        edgecolor=output_color, facecolor=output_color, alpha=0.3, linewidth=2)
            ax.add_patch(output_box)
            ax.text(2.5, 2.5, 'Predictions', ha='center', va='center', fontsize=11, fontweight='bold')

            # Arrow from hidden to output
            arrow4 = FancyArrowPatch((2.5, 3.5), (2.5, 3), arrowstyle='->', mutation_scale=20, linewidth=2, color='gray')
            ax.add_patch(arrow4)

            # Problem annotation
            problem_box = FancyBboxPatch((4.5, 4.5), 3, 1.5, boxstyle="round,pad=0.1",
                                          edgecolor=problem_color, facecolor='white', linestyle='--', linewidth=2)
            ax.add_patch(problem_box)
            ax.text(6, 5.25, '⚠️ PROBLEM:\nDomains may be\nignored by model!', ha='center', va='center',
                    fontsize=10, color=problem_color, fontweight='bold')

            ax.set_title('Current Model\n(Simple Concatenation)', fontsize=14, fontweight='bold', pad=20)

        else:
            # Improved model: separate pathways with attention

            # ESM pathway
            esm_proj_box = FancyBboxPatch((0.5, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                                          edgecolor=esm_color, facecolor=esm_color, alpha=0.2, linewidth=2)
            ax.add_patch(esm_proj_box)
            ax.text(1.5, 5.9, 'ESM\nProjection', ha='center', va='center', fontsize=10)

            # Domain pathway
            domain_encoder_box = FancyBboxPatch((3.5, 5.5), 2, 0.8, boxstyle="round,pad=0.1",
                                                edgecolor=domain_color, facecolor=domain_color, alpha=0.2, linewidth=2)
            ax.add_patch(domain_encoder_box)
            ax.text(4.5, 5.9, 'Domain\nEncoder', ha='center', va='center', fontsize=10)

            # Arrows from inputs to pathways
            arrow1 = FancyArrowPatch((1.5, 7), (1.5, 6.3), arrowstyle='->', mutation_scale=20, linewidth=2, color=esm_color)
            arrow2 = FancyArrowPatch((4.5, 7), (4.5, 6.3), arrowstyle='->', mutation_scale=20, linewidth=2, color=domain_color)
            ax.add_patch(arrow1)
            ax.add_patch(arrow2)

            # Attention mechanism
            attention_box = FancyBboxPatch((1, 4), 4, 1, boxstyle="round,pad=0.1",
                                          edgecolor=solution_color, facecolor=solution_color, alpha=0.3, linewidth=3)
            ax.add_patch(attention_box)
            ax.text(3, 4.5, 'Feature Fusion\nwith Attention', ha='center', va='center', fontsize=11, fontweight='bold')

            # Arrows from pathways to attention
            arrow3 = FancyArrowPatch((1.5, 5.5), (2, 5), arrowstyle='->', mutation_scale=20, linewidth=2, color=esm_color)
            arrow4 = FancyArrowPatch((4.5, 5.5), (4, 5), arrowstyle='->', mutation_scale=20, linewidth=2, color=domain_color)
            ax.add_patch(arrow3)
            ax.add_patch(arrow4)

            # Attention weights display
            weight_box = FancyBboxPatch((5.5, 3.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                                        edgecolor=solution_color, facecolor='white', linestyle='-', linewidth=2)
            ax.add_patch(weight_box)
            ax.text(6.75, 4.5, '📊 Visible Weights:', ha='center', va='center', fontsize=9, fontweight='bold')
            ax.text(6.75, 4.1, 'ESM: 0.70', ha='center', va='center', fontsize=9, color=esm_color, fontweight='bold')
            ax.text(6.75, 3.8, 'Domain: 0.30', ha='center', va='center', fontsize=9, color=domain_color, fontweight='bold')

            # Arrow from attention to weights
            arrow_weight = FancyArrowPatch((5, 4.5), (5.5, 4.5), arrowstyle='->', mutation_scale=15,
                                          linewidth=1.5, color=solution_color, linestyle='--')
            ax.add_patch(arrow_weight)

            # Hidden layers
            hidden_box = FancyBboxPatch((1.5, 2.5), 2, 0.8, boxstyle="round,pad=0.1",
                                        edgecolor='gray', facecolor='lightgray', alpha=0.5, linewidth=2)
            ax.add_patch(hidden_box)
            ax.text(2.5, 2.9, 'Hidden Layers', ha='center', va='center', fontsize=10)

            # Arrow from attention to hidden
            arrow5 = FancyArrowPatch((3, 4), (2.5, 3.3), arrowstyle='->', mutation_scale=20, linewidth=2, color='gray')
            ax.add_patch(arrow5)

            # Output
            output_box = FancyBboxPatch((1.5, 1), 2, 0.8, boxstyle="round,pad=0.1",
                                        edgecolor=output_color, facecolor=output_color, alpha=0.3, linewidth=2)
            ax.add_patch(output_box)
            ax.text(2.5, 1.4, 'Predictions', ha='center', va='center', fontsize=11, fontweight='bold')

            # Arrow from hidden to output
            arrow6 = FancyArrowPatch((2.5, 2.5), (2.5, 1.8), arrowstyle='->', mutation_scale=20, linewidth=2, color='gray')
            ax.add_patch(arrow6)

            # Solution annotation
            solution_box = FancyBboxPatch((0.2, 0), 5.6, 0.7, boxstyle="round,pad=0.1",
                                          edgecolor=solution_color, facecolor='white', linestyle='-', linewidth=2)
            ax.add_patch(solution_box)
            ax.text(3, 0.35, '✅ SOLUTION: Explicit pathways ensure both features are used!',
                    ha='center', va='center', fontsize=10, color=solution_color, fontweight='bold')

            ax.set_title('Improved Model\n(Explicit Feature Fusion)', fontsize=14, fontweight='bold', pad=20)

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 9)
        ax.axis('off')

    # Draw both architectures
    draw_architecture(ax1, is_improved=False)
    draw_architecture(ax2, is_improved=True)

    plt.suptitle('Domain Feature Usage: Current vs Improved Model', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('model_architecture_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_architecture_comparison.png")
    plt.show()


def create_diagnostic_workflow():
    """Create workflow diagram showing diagnostic process"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Colors
    step_color = '#3498DB'
    check_color = '#F39C12'
    good_color = '#2ECC71'
    bad_color = '#E74C3C'

    y_pos = 9

    # Step 1: Run diagnostics
    step1 = FancyBboxPatch((1, y_pos), 10, 1, boxstyle="round,pad=0.1",
                           edgecolor=step_color, facecolor=step_color, alpha=0.3, linewidth=3)
    ax.add_patch(step1)
    ax.text(6, y_pos + 0.5, '1️⃣  Run ablation_test(model, val_loader, device)',
            ha='center', va='center', fontsize=12, fontweight='bold')
    y_pos -= 1.5

    # Arrow
    arrow1 = FancyArrowPatch((6, y_pos + 1), (6, y_pos + 0.5), arrowstyle='->', mutation_scale=30,
                            linewidth=3, color='gray')
    ax.add_patch(arrow1)
    y_pos -= 0.5

    # Check result
    check = FancyBboxPatch((2, y_pos), 8, 1, boxstyle="round,pad=0.1",
                           edgecolor=check_color, facecolor=check_color, alpha=0.2, linewidth=3)
    ax.add_patch(check)
    ax.text(6, y_pos + 0.5, '📊 Check: mean_diff value',
            ha='center', va='center', fontsize=11, fontweight='bold')
    y_pos -= 1.5

    # Branch: Good result
    arrow_good = FancyArrowPatch((6, y_pos + 1), (3, y_pos + 0.5), arrowstyle='->', mutation_scale=25,
                                linewidth=2.5, color=good_color)
    ax.add_patch(arrow_good)

    good_box = FancyBboxPatch((0.5, y_pos - 1), 5, 1.5, boxstyle="round,pad=0.1",
                              edgecolor=good_color, facecolor=good_color, alpha=0.2, linewidth=3)
    ax.add_patch(good_box)
    ax.text(3, y_pos - 0.25, '✅ mean_diff > 0.01\nDomains ARE used!\nContinue training.',
            ha='center', va='center', fontsize=10, fontweight='bold', color=good_color)

    # Branch: Bad result
    arrow_bad = FancyArrowPatch((6, y_pos + 1), (9, y_pos + 0.5), arrowstyle='->', mutation_scale=25,
                               linewidth=2.5, color=bad_color)
    ax.add_patch(arrow_bad)

    bad_box = FancyBboxPatch((6.5, y_pos - 1), 5, 1.5, boxstyle="round,pad=0.1",
                             edgecolor=bad_color, facecolor=bad_color, alpha=0.2, linewidth=3)
    ax.add_patch(bad_box)
    ax.text(9, y_pos - 0.25, '⚠️ mean_diff < 0.01\nDomains NOT used!\nApply fixes below ↓',
            ha='center', va='center', fontsize=10, fontweight='bold', color=bad_color)

    y_pos -= 2.5

    # Arrow from bad to fixes
    arrow_fix = FancyArrowPatch((9, y_pos + 1), (9, y_pos + 0.5), arrowstyle='->', mutation_scale=25,
                               linewidth=2.5, color=bad_color)
    ax.add_patch(arrow_fix)
    y_pos -= 0.5

    # Fix 1
    fix1 = FancyBboxPatch((6.5, y_pos), 5, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='#9B59B6', facecolor='#9B59B6', alpha=0.2, linewidth=2)
    ax.add_patch(fix1)
    ax.text(9, y_pos + 0.4, 'Fix 1: Use ImprovedProteinClassifier',
            ha='center', va='center', fontsize=10, fontweight='bold')
    y_pos -= 1

    # Fix 2
    fix2 = FancyBboxPatch((6.5, y_pos), 5, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='#9B59B6', facecolor='#9B59B6', alpha=0.2, linewidth=2)
    ax.add_patch(fix2)
    ax.text(9, y_pos + 0.4, 'Fix 2: Increase domain_output_dim to 512',
            ha='center', va='center', fontsize=10, fontweight='bold')
    y_pos -= 1

    # Fix 3
    fix3 = FancyBboxPatch((6.5, y_pos), 5, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='#9B59B6', facecolor='#9B59B6', alpha=0.2, linewidth=2)
    ax.add_patch(fix3)
    ax.text(9, y_pos + 0.4, 'Fix 3: Add FeatureUsageMonitor',
            ha='center', va='center', fontsize=10, fontweight='bold')
    y_pos -= 1.5

    # Retrain
    retrain = FancyBboxPatch((6.5, y_pos), 5, 0.8, boxstyle="round,pad=0.1",
                            edgecolor=step_color, facecolor=step_color, alpha=0.3, linewidth=2)
    ax.add_patch(retrain)
    ax.text(9, y_pos + 0.4, '🔄 Retrain model with fixes',
            ha='center', va='center', fontsize=11, fontweight='bold')
    y_pos -= 1.2

    # Arrow to recheck
    arrow_recheck = FancyArrowPatch((9, y_pos + 0.8), (9, y_pos + 0.3), arrowstyle='->', mutation_scale=25,
                                   linewidth=2.5, color='gray')
    ax.add_patch(arrow_recheck)
    y_pos -= 0.3

    # Verify
    verify = FancyBboxPatch((6.5, y_pos), 5, 0.8, boxstyle="round,pad=0.1",
                           edgecolor=good_color, facecolor=good_color, alpha=0.3, linewidth=3)
    ax.add_patch(verify)
    ax.text(9, y_pos + 0.4, '✅ Verify: mean_diff improved!',
            ha='center', va='center', fontsize=11, fontweight='bold', color=good_color)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Domain Feature Diagnostic Workflow', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('diagnostic_workflow.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: diagnostic_workflow.png")
    plt.show()


if __name__ == "__main__":
    print("Creating visualizations...\n")
    create_architecture_comparison()
    print()
    create_diagnostic_workflow()
    print("\n✅ All visualizations created!")
    print("   - model_architecture_comparison.png")
    print("   - diagnostic_workflow.png")
