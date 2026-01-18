#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAFA-6 Multi-Ontology Training - V10 WITH DOMAIN FEATURES + DIAGNOSTICS
=================================================================================
Integrates InterPro/Signature domain annotations from unique_superset.tsv
Key: Maps Protein_accession to pid, uses InterPro_accession (fallback: Signature_accession)

UPDATED: Added domain feature diagnostics to verify domain usage
"""

import os
import gc
import random
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# ===== DOMAIN DIAGNOSTICS =====
# NEW: Import diagnostic tools to verify domain feature usage
try:
    from domain_diagnostics import ablation_test, compute_feature_importance
    DIAGNOSTICS_AVAILABLE = True
    print("✓ Domain diagnostics loaded successfully")
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    print("⚠ Domain diagnostics not available (domain_diagnostics.py not found)")

# ===== REPRODUCIBILITY =====
SEED = 999
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== CONFIGURATION =====
CONFIG = {
    'P': {
        'hidden_dims': [1024, 768, 512, 256],
        'dropout': 0.15,
        'lr': 5e-4,
        'epochs': 90,
        'batch_size': 64,
        'accumulation_steps': 4,
        'lr_warmup_ratio': 0.1,
        'use_class_weights': True,
        'class_weight_smoothing': 0.1,
    },
    'C': {
        'hidden_dims': [1024, 768, 512, 256],
        'dropout': 0.2,
        'lr': 1e-3,
        'epochs': 60,
        'batch_size': 128,
        'accumulation_steps': 2,
        'lr_warmup_ratio': 0.1,
        'use_class_weights': True,
        'class_weight_smoothing': 0.05,
    },
    'F': {
        'hidden_dims': [768, 512, 256],
        'dropout': 0.15,
        'lr': 1e-3,
        'epochs': 60,
        'batch_size': 128,
        'accumulation_steps': 2,
        'lr_warmup_ratio': 0.1,
        'use_class_weights': True,
        'class_weight_smoothing': 0.05,
    },
    'weight_decay': 3e-5,
    'normalize_embeddings': True,
    'label_smoothing': 0.03,
    'patience': 25,
    'grad_clip': 1.0,
    'use_ema': True,
    'ema_decay': 0.999,
    'val_species_ratio': 0.10,
    'min_species_proteins': 5,
    # Domain feature settings
    'use_domains': True,
    'domain_embed_dim': 64,
    'domain_output_dim': 256,
}

# ===== PATHS =====
IS_LOCAL = not os.path.isdir("/kaggle/input")

TRAIN_TERMS_PATH = '/kaggle/input/cafa-6-protein-function-prediction/Train/train_terms.tsv'
TRAIN_SEQUENCES_PATH = '/kaggle/input/cafa-6-protein-function-prediction/Train/train_sequences.fasta'
TEST_SEQUENCES_PATH = '/kaggle/input/cafa-6-protein-function-prediction/Test/testsuperset.fasta'
TRAIN_TAXONOMY_PATH = '/kaggle/input/cafa-6-protein-function-prediction/Train/train_taxonomy.tsv'
IA_PATH = '/kaggle/input/cafa-6-protein-function-prediction/IA.tsv'
PROT_EMBEDS = '/kaggle/input/esm-650m-embeds/esm-650m-embeds/protein_embeddings.npy'
PIDS = '/kaggle/input/esm-650m-embeds/esm-650m-embeds/protein_ids.csv'
OBO_PATH = '/kaggle/input/cafa-6-protein-function-prediction/Train/go-basic.obo'
GOA_PATH = '/kaggle/input/protein-go-annotations/goa_uniprot_all.csv'
TAXID_DIST_PATH = '/kaggle/input/tax-id-distances-cafa6/taxid_symmetric.csv'
DOMAIN_PATH = '/kaggle/input/unique-domains-for-proteins-cafa6/unique_superset.tsv'

if IS_LOCAL:
    TRAIN_TERMS_PATH = './data/cafa-6-protein-function-prediction/Train/train_terms.tsv'
    TRAIN_SEQUENCES_PATH = './data/cafa-6-protein-function-prediction/Train/train_sequences.fasta'
    TEST_SEQUENCES_PATH = './data/cafa-6-protein-function-prediction/Test/testsuperset.fasta'
    TRAIN_TAXONOMY_PATH = './data/cafa-6-protein-function-prediction/Train/train_taxonomy.tsv'
    IA_PATH = './data/cafa-6-protein-function-prediction/IA.tsv'
    PROT_EMBEDS = './data/esm-650m-embeds/protein_embeddings.npy'
    PIDS = './data/esm-650m-embeds/protein_ids.csv'
    OBO_PATH = './data/cafa-6-protein-function-prediction/Train/go-basic.obo'
    GOA_PATH = './data/cafa-6-protein-function-prediction/goa_uniprot_all.csv'
    TAXID_DIST_PATH = './taxid_symmetric.csv'
    DOMAIN_PATH = './data/unique_superset.tsv'

# ===== LOAD EMBEDDINGS =====
print("Loading embeddings...")
protein_ids = pd.read_csv(PIDS)["protein_id"].tolist()
embeddings = np.load(PROT_EMBEDS)

if CONFIG['normalize_embeddings']:
    print("Normalizing embeddings (Z-score)...")
    mean = np.mean(embeddings, axis=0, keepdims=True)
    std = np.std(embeddings, axis=0, keepdims=True)
    embeddings = (embeddings - mean) / (std + 1e-8)

embeddings_dict = {pid: emb for pid, emb in zip(protein_ids, embeddings)}
EMBED_DIM = embeddings.shape[1]
print(f"Loaded {len(protein_ids)} embeddings of dim {EMBED_DIM}")


# ===== LOAD DOMAIN FEATURES =====
def load_domain_features(domain_path):
    """Load and process domain annotations from unique_superset.tsv"""
    print(f"\nLoading domain annotations from {domain_path}...")

    if not os.path.exists(domain_path):
        print(f"  [WARNING] Domain file not found, proceeding without domain features")
        return {}, {}, 0

    df = pd.read_csv(domain_path, sep='\t', on_bad_lines='skip')
    print(f"  Raw annotations: {len(df)} rows, {df['Protein_accession'].nunique()} proteins")

    # Create domain ID: InterPro_accession if valid, else Signature_accession
    df['domain_id'] = df.apply(
        lambda row: row['InterPro_accession']
        if pd.notna(row['InterPro_accession']) and row['InterPro_accession'] != '-'
        else row['Signature_accession'],
        axis=1
    )

    # Filter invalid domains
    df = df[df['domain_id'].notna() & (df['domain_id'] != '-') & (df['domain_id'] != '')]
    print(f"  After filtering: {len(df)} annotations")

    # Build protein -> domains mapping
    protein_domains = defaultdict(list)
    for _, row in df.iterrows():
        protein_domains[row['Protein_accession']].append({
            'domain_id': row['domain_id'],
            'start': row['Start'],
            'end': row['End'],
            'length': row['Length'],
            'seq_length': row['Sequence_length'],
            'score': row['Score'] if pd.notna(row['Score']) else 0.0,
        })

    # Build domain vocabulary
    all_domains = sorted(set(d['domain_id'] for domains in protein_domains.values() for d in domains))
    domain_to_idx = {d: i+1 for i, d in enumerate(all_domains)}  # 0 = padding
    num_domains = len(domain_to_idx) + 1  # +1 for padding

    print(f"  Unique domains: {len(all_domains)}")
    print(f"  Proteins with domains: {len(protein_domains)}")

    # Create multi-hot encodings
    pid_to_domain_multihot = {}
    for pid, domains in protein_domains.items():
        multihot = np.zeros(num_domains, dtype=np.float32)
        for d in domains:
            idx = domain_to_idx[d['domain_id']]
            # Weight by coverage (capped at 1.0)
            coverage = min(1.0, d['length'] / max(d['seq_length'], 1) * 2)
            multihot[idx] = max(multihot[idx], coverage)
        pid_to_domain_multihot[pid] = multihot

    return protein_domains, pid_to_domain_multihot, num_domains


protein_domains, pid_to_domain_multihot, NUM_DOMAINS = load_domain_features(DOMAIN_PATH)
USE_DOMAINS = CONFIG['use_domains'] and NUM_DOMAINS > 0
print(f"Domain features: {'ENABLED' if USE_DOMAINS else 'DISABLED'} ({NUM_DOMAINS} domain types)")


# ===== HELPER FUNCTIONS (same as before) =====
def parse_fasta(fasta_file):
    sequences = {}
    current_id, current_seq = None, []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                parts = line[1:].split('|')
                current_id = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    return sequences


print("\nLoading sequences and annotations...")
train_terms_df = pd.read_csv(TRAIN_TERMS_PATH, sep='\t')
train_sequences = parse_fasta(TRAIN_SEQUENCES_PATH)
test_sequences = parse_fasta(TEST_SEQUENCES_PATH)
print(f"Train sequences: {len(train_sequences)}, Test sequences: {len(test_sequences)}")

# Check domain coverage
train_with_domains = sum(1 for p in train_sequences if p in pid_to_domain_multihot)
test_with_domains = sum(1 for p in test_sequences if p in pid_to_domain_multihot)
print(f"Domain coverage - Train: {train_with_domains}/{len(train_sequences)} ({100*train_with_domains/len(train_sequences):.1f}%)")
print(f"Domain coverage - Test: {test_with_domains}/{len(test_sequences)} ({100*test_with_domains/len(test_sequences):.1f}%)")

# Load taxonomy
print("\nLoading taxonomy...")
train_taxonomy = pd.read_csv(TRAIN_TAXONOMY_PATH, sep='\t', names=['protein', 'taxon'])
pid_to_taxon = dict(zip(train_taxonomy['protein'], train_taxonomy['taxon']))

# Load IA weights
print("Loading IA weights...")
ia_df = pd.read_csv(IA_PATH, sep='\t', names=['term', 'ia'])
ia_dict = dict(zip(ia_df['term'], ia_df['ia']))


def parse_obo(obo_file):
    go_parents, go_children, go_namespace = {}, {}, {}
    current_term, is_obsolete, current_namespace = None, False, None

    with open(obo_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_term, is_obsolete, current_namespace = None, False, None
            elif line.startswith("id: GO:"):
                current_term = line.split("id: ")[1]
                go_parents[current_term] = []
                go_children.setdefault(current_term, [])
            elif line.startswith("namespace:"):
                current_namespace = line.split("namespace: ")[1]
                if current_term:
                    go_namespace[current_term] = current_namespace
            elif line.startswith("is_obsolete: true"):
                is_obsolete = True
            elif line.startswith("is_a:") and current_term and not is_obsolete:
                parent = line.split("is_a: ")[1].split(" !")[0].strip()
                go_parents[current_term].append(parent)
                go_children.setdefault(parent, []).append(current_term)
            elif line.startswith("relationship: part_of") and current_term and not is_obsolete:
                parent = line.split("part_of ")[1].split(" !")[0].strip()
                go_parents[current_term].append(parent)
                go_children.setdefault(parent, []).append(current_term)

    return go_parents, go_children, go_namespace


go_parents, go_children, go_namespace = parse_obo(OBO_PATH)
print(f"Loaded {len(go_parents)} GO terms")


# ===== GOA INTEGRATION (same as before) =====
def get_all_descendants(term, go_children, cache=None):
    if cache is None:
        cache = {}
    if term in cache:
        return cache[term]
    descendants = set()
    stack = [term]
    while stack:
        cur = stack.pop()
        for child in go_children.get(cur, []):
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    cache[term] = descendants
    return descendants


def load_goa_and_build_negative_keys(goa_path, go_children):
    if not os.path.exists(goa_path):
        return set(), None

    print(f"Loading GOA annotations...")
    goa_df = pd.read_csv(goa_path).drop_duplicates()

    negative_annots = goa_df[goa_df['qualifier'].str.contains('NOT', na=False)]
    negative_annots = negative_annots[['protein_id', 'go_term']].drop_duplicates()
    negative_by_protein = negative_annots.groupby('protein_id')['go_term'].apply(list).to_dict()

    desc_cache = {}
    negative_keys = set()
    for protein, terms in tqdm(negative_by_protein.items(), desc="Propagating negatives"):
        all_negative = set(terms)
        for term in terms:
            all_negative |= get_all_descendants(term, go_children, desc_cache)
        for term in all_negative:
            negative_keys.add(f"{protein}_{term}")

    positive_annots = goa_df[~goa_df['qualifier'].str.contains('NOT', na=False)]
    positive_annots = positive_annots[['protein_id', 'go_term']].drop_duplicates()
    positive_annots['score'] = 1.0
    positive_annots['pred_key'] = positive_annots['protein_id'].astype(str) + '_' + positive_annots['go_term'].astype(str)
    positive_annots = positive_annots[~positive_annots['pred_key'].isin(negative_keys)]

    return negative_keys, positive_annots


negative_keys, goa_positive_df = load_goa_and_build_negative_keys(GOA_PATH, go_children)
all_pid_to_taxon = dict(pid_to_taxon)


# ===== TAXONOMY (condensed) =====
def load_taxonomy_distances(path):
    if not os.path.exists(path):
        return None, set()
    df = pd.read_csv(path, index_col=0)
    df.index, df.columns = df.index.astype(int), df.columns.astype(int)
    return df, set(df.index)


def build_species_go_profiles(train_terms_df, pid_to_taxon, goa_df=None):
    species_terms = {}
    for _, row in train_terms_df.iterrows():
        pid, term = row['EntryID'], row['term']
        if pid in pid_to_taxon:
            taxon = pid_to_taxon[pid]
            species_terms.setdefault(taxon, Counter())[term] += 1

    species_profiles = {}
    for taxon, term_counts in species_terms.items():
        total = sum(term_counts.values())
        species_profiles[taxon] = {term: count/total for term, count in term_counts.items()}
    return species_profiles


taxid_dist_df, available_taxa = load_taxonomy_distances(TAXID_DIST_PATH)
species_profiles = None


def create_species_aware_split(pid_list, pid_to_taxon, val_ratio=0.10, min_proteins=5, seed=SEED):
    np.random.seed(seed)
    pids_with_taxon = [p for p in pid_list if p in pid_to_taxon]
    pids_without_taxon = [p for p in pid_list if p not in pid_to_taxon]

    taxon_to_pids = {}
    for pid in pids_with_taxon:
        taxon_to_pids.setdefault(pid_to_taxon[pid], []).append(pid)

    eligible_taxa = [t for t, pids in taxon_to_pids.items() if len(pids) >= min_proteins]
    np.random.shuffle(eligible_taxa)

    total_proteins = len(pids_with_taxon)
    target_val_size = int(total_proteins * val_ratio)

    val_taxa, val_size = [], 0
    for taxon in eligible_taxa:
        species_size = len(taxon_to_pids[taxon])
        if species_size > total_proteins * 0.05:
            continue
        if val_size + species_size <= target_val_size * 1.2:
            val_taxa.append(taxon)
            val_size += species_size
        if val_size >= target_val_size:
            break

    val_taxa_set = set(val_taxa)
    train_pids = [p for p in pids_with_taxon if pid_to_taxon[p] not in val_taxa_set] + pids_without_taxon
    val_pids = [p for p in pids_with_taxon if pid_to_taxon[p] in val_taxa_set]

    np.random.shuffle(train_pids)
    np.random.shuffle(val_pids)

    print(f"  Train: {len(train_pids)}, Val: {len(val_pids)}")
    return train_pids, val_pids


# ===== CLASS WEIGHTS & EMA =====
def compute_class_weights(labels, num_classes, smoothing=0.1):
    class_counts = np.array(labels.sum(axis=0)).flatten() if hasattr(labels, 'toarray') else labels.sum(axis=0)
    weights = 1.0 / (class_counts + smoothing)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(np.clip(weights, 0.1, 10.0), dtype=torch.float32)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model, self.decay = model, decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===== FMAX COMPUTATION =====
def compute_fmax_adaptive(preds, labels, ia_weights=None, use_ia=True):
    if use_ia and ia_weights is not None:
        ia_weights = torch.tensor(ia_weights, dtype=torch.float32).to(preds.device) if not isinstance(ia_weights, torch.Tensor) else ia_weights.to(preds.device)

    def compute_f1(t):
        binary = (preds >= t).float()
        if use_ia and ia_weights is not None:
            tp = (binary * labels * ia_weights).sum()
            pred_pos, actual_pos = (binary * ia_weights).sum(), (labels * ia_weights).sum()
        else:
            tp, pred_pos, actual_pos = (binary * labels).sum(), binary.sum(), labels.sum()
        prec, rec = tp / (pred_pos + 1e-8), tp / (actual_pos + 1e-8)
        return (2 * prec * rec / (prec + rec + 1e-8)).item(), prec.item(), rec.item()

    # Coarse then fine search
    best_t, best_fmax = 0.3, 0
    for t in np.arange(0.05, 0.95, 0.1):
        f1, _, _ = compute_f1(t)
        if f1 > best_fmax:
            best_fmax, best_t = f1, t

    best_prec, best_rec = 0, 0
    for t in np.arange(max(0.01, best_t - 0.15), min(0.99, best_t + 0.15), 0.01):
        f1, prec, rec = compute_f1(t)
        if f1 > best_fmax:
            best_fmax, best_t, best_prec, best_rec = f1, t, prec, rec

    return best_fmax, best_t, best_prec, best_rec


# ===== LOSS FUNCTION =====
class AsymmetricLossWithWeights(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, label_smoothing=0.03, class_weights=None):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps, self.label_smoothing = gamma_neg, gamma_pos, clip, eps, label_smoothing
        self.register_buffer('class_weights', class_weights)

    def forward(self, x, y):
        if self.label_smoothing > 0:
            y = y * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        xs_pos = torch.sigmoid(x)
        xs_neg = (1 - xs_pos + self.clip).clamp(max=1) if self.clip else 1 - xs_pos

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        pt = xs_pos * y + xs_neg * (1 - y)
        gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        loss = torch.pow(1 - pt, gamma) * (los_pos + los_neg)

        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        return -loss.sum(dim=1).mean()


# ===== MODEL COMPONENTS =====
class FeatureGating(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 64)
        self.gate = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.fc = nn.Sequential(nn.Linear(dim, hidden, bias=False), nn.ReLU(True), nn.Linear(hidden, dim, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, expansion=2, dropout=0.1, use_se=True):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))
        self.se = SEBlock(dim) if use_se else nn.Identity()

    def forward(self, x):
        return x + self.se(self.ffn(self.norm(x)))


class DomainEncoder(nn.Module):
    """Encodes domain multi-hot vectors into dense representations."""
    def __init__(self, num_domains, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_domains, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, domain_multihot):
        return self.encoder(domain_multihot)


# ===== MAIN MODEL =====
class ProteinClassifierWithDomains(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims, dropout=0.2,
                 num_domains=0, domain_output_dim=256, use_domains=True):
        super().__init__()
        self.use_domains = use_domains and num_domains > 0

        # Domain encoder
        if self.use_domains:
            self.domain_encoder = DomainEncoder(num_domains, domain_output_dim)
            combined_dim = input_dim + domain_output_dim
            print(f"    Model input: ESM({input_dim}) + Domains({domain_output_dim}) = {combined_dim}")
        else:
            combined_dim = input_dim

        # Input processing
        self.input_norm = nn.LayerNorm(combined_dim)
        self.feature_gate = FeatureGating(combined_dim, reduction=4)
        self.input_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Encoder blocks
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.blocks.append(ResidualBlock(hidden_dims[i], expansion=2, dropout=dropout))
            self.transitions.append(nn.Sequential(
                nn.LayerNorm(hidden_dims[i]),
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

        self.blocks.append(ResidualBlock(hidden_dims[-1], expansion=2, dropout=dropout))
        self.final_norm = nn.LayerNorm(hidden_dims[-1])

        # Multi-scale heads
        self.output_heads = nn.ModuleList([nn.Linear(hidden_dims[-1], num_classes)])
        if len(hidden_dims) > 1:
            self.output_heads.append(nn.Linear(hidden_dims[-2], num_classes))
            self.intermediate_proj = nn.Sequential(nn.LayerNorm(hidden_dims[-2]), nn.Dropout(dropout))

        self.head_weights = nn.Parameter(torch.ones(len(self.output_heads)) / len(self.output_heads))
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
            x = torch.cat([x, domain_feat], dim=-1)

        x = self.input_norm(x)
        x = self.feature_gate(x)
        x = self.input_proj(x)

        intermediates = []
        for block, transition in zip(self.blocks[:-1], self.transitions):
            x = block(x)
            intermediates.append(x)
            x = transition(x)

        x = self.blocks[-1](x)
        x = self.final_norm(x)

        weights = F.softmax(self.head_weights, dim=0)
        output = weights[0] * self.output_heads[0](x)

        if len(self.output_heads) > 1 and intermediates:
            output = output + weights[1] * self.output_heads[1](self.intermediate_proj(intermediates[-1]))

        return output


# ===== DATASETS =====
class ProtDatasetWithDomains(Dataset):
    def __init__(self, pids, labels, embeddings_dict, pid_to_domain_multihot, num_domains):
        self.pids, self.labels = pids, labels
        self.embeddings_dict, self.pid_to_domain_multihot = embeddings_dict, pid_to_domain_multihot
        self.num_domains = num_domains

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        embed = self.embeddings_dict[pid]
        label = self.labels[idx].toarray().flatten() if hasattr(self.labels, 'toarray') else self.labels[idx]
        domain = self.pid_to_domain_multihot.get(pid, np.zeros(self.num_domains, dtype=np.float32))

        return torch.from_numpy(embed).float(), torch.from_numpy(domain).float(), torch.tensor(label, dtype=torch.float32)


class TestDataSetWithDomains(Dataset):
    def __init__(self, pids, embeddings_dict, pid_to_domain_multihot, num_domains):
        self.pids, self.embeddings_dict = pids, embeddings_dict
        self.pid_to_domain_multihot, self.num_domains = pid_to_domain_multihot, num_domains

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        embed = self.embeddings_dict[pid]
        domain = self.pid_to_domain_multihot.get(pid, np.zeros(self.num_domains, dtype=np.float32))
        return torch.from_numpy(embed).float(), torch.from_numpy(domain).float()


# ===== TRAINING =====
def train_model(aspect, model, train_loader, valid_loader, ia_weights_tensor, config, class_weights=None):
    model = model.to(device)
    ia_weights_tensor = ia_weights_tensor.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    loss_fn = AsymmetricLossWithWeights(
        gamma_neg=4, gamma_pos=1, clip=0.05,
        label_smoothing=CONFIG['label_smoothing'],
        class_weights=class_weights
    )

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=CONFIG['weight_decay'])

    accum_steps = config.get('accumulation_steps', 1)
    steps_per_epoch = len(train_loader) // accum_steps
    total_steps = steps_per_epoch * config['epochs']
    warmup_steps = int(total_steps * config.get('lr_warmup_ratio', 0.1))

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ema = EMA(model, CONFIG.get('ema_decay', 0.999)) if CONFIG.get('use_ema', True) else None
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_fmax, best_state, best_t, patience = 0, None, 0.1, 0
    history = {'train_loss': [], 'val_loss': [], 'fmax_weighted': [], 'fmax_unweighted': [], 'threshold': [], 'precision': [], 'recall': [], 'lr': []}

    print(f"  Steps: {total_steps}, Warmup: {warmup_steps}, Accum: {accum_steps}")

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (X, domain, y) in enumerate(train_loader):
            X, domain, y = X.to(device), domain.to(device), y.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(X, domain_multihot=domain)
                loss = loss_fn(outputs, y) / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.get('grad_clip', 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                if ema:
                    ema.update()

            train_loss += loss.item() * accum_steps * X.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        if ema:
            ema.apply_shadow()

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, domain, y in valid_loader:
                X, domain, y = X.to(device), domain.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(X, domain_multihot=domain)
                    loss = loss_fn(outputs, y)
                val_loss += loss.item() * X.size(0)
                all_preds.append(torch.sigmoid(outputs))
                all_labels.append(y)

        if ema:
            ema.restore()

        avg_val_loss = val_loss / len(valid_loader.dataset)
        all_preds, all_labels = torch.cat(all_preds), torch.cat(all_labels)

        fmax_w, t_w, prec_w, rec_w = compute_fmax_adaptive(all_preds, all_labels, ia_weights_tensor, use_ia=True)
        fmax_uw, _, _, _ = compute_fmax_adaptive(all_preds, all_labels, None, use_ia=False)

        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['fmax_weighted'].append(fmax_w)
        history['fmax_unweighted'].append(fmax_uw)
        history['threshold'].append(t_w)
        history['precision'].append(prec_w)
        history['recall'].append(rec_w)
        history['lr'].append(lr)

        marker = ""
        if fmax_w > best_fmax:
            best_fmax, best_t = fmax_w, t_w
            if ema:
                ema.apply_shadow()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ema:
                ema.restore()
            patience = 0
            marker = " ⭐"
        else:
            patience += 1

        print(f"[{aspect}] Ep {epoch:2d} | LR {lr:.1e} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
              f"Fmax(IA) {fmax_w:.4f} (t={t_w:.2f}) | Fmax {fmax_uw:.4f} | P:{prec_w:.3f} R:{rec_w:.3f}{marker}")

        if patience >= CONFIG['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
        print(f"[{aspect}] Best: Fmax(IA)={best_fmax:.4f} at t={best_t:.2f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0,0].plot(history['train_loss'], label='Train'); axes[0,0].plot(history['val_loss'], label='Val')
    axes[0,0].set_title(f'{aspect} Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(history['fmax_weighted'], 'g-', label='IA-weighted'); axes[0,1].plot(history['fmax_unweighted'], 'b--', label='Unweighted')
    axes[0,1].axhline(best_fmax, color='r', linestyle=':'); axes[0,1].set_title(f'{aspect} Fmax'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
    axes[0,2].plot(history['lr'], 'purple'); axes[0,2].set_title(f'{aspect} LR'); axes[0,2].set_yscale('log'); axes[0,2].grid(True, alpha=0.3)
    axes[1,0].plot(history['threshold'], 'purple'); axes[1,0].set_title(f'{aspect} Threshold'); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(history['precision'], 'b-', label='P'); axes[1,1].plot(history['recall'], 'orange', label='R')
    axes[1,1].set_title(f'{aspect} P/R'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
    if hasattr(model, 'head_weights'):
        w = F.softmax(model.head_weights.detach().cpu(), dim=0).numpy()
        axes[1,2].bar(range(len(w)), w); axes[1,2].set_title(f'{aspect} Head Weights')
    plt.suptitle(f'{aspect} Training (V10 + Domains)', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{aspect}_training_v10.png', dpi=150); plt.show()

    return model, best_fmax, best_t


# ===== MAIN TRAINING LOOP =====
ASPECTS = ['F', 'C', 'P']
aspect_models, aspect_mlbs, aspect_thresholds, aspect_ia_weights = {}, {}, {}, {}


def train_all():
    global species_profiles
    species_profiles = build_species_go_profiles(train_terms_df, pid_to_taxon, goa_positive_df)
    results = []

    for aspect in ASPECTS:
        print(f"\n{'='*80}")
        name = {'F': 'Molecular Function', 'C': 'Cellular Component', 'P': 'Biological Process'}[aspect]
        print(f"TRAINING {name} ({aspect}) WITH DOMAIN FEATURES")
        print('='*80)

        config = CONFIG[aspect]
        aspect_df = train_terms_df[train_terms_df['aspect'] == aspect]
        protein_2_terms = aspect_df.groupby('EntryID')['term'].apply(list).to_dict()

        pid_all = [p for p in train_sequences if p in embeddings_dict and p in protein_2_terms]
        n_dom = sum(1 for p in pid_all if p in pid_to_domain_multihot)
        print(f"  Proteins: {len(pid_all)}, with domains: {n_dom} ({100*n_dom/len(pid_all):.1f}%)")

        train_pids, val_pids = create_species_aware_split(pid_all, pid_to_taxon, CONFIG['val_species_ratio'], CONFIG['min_species_proteins'])

        all_terms = set(t for p in train_pids for t in protein_2_terms[p])
        mlb = MultiLabelBinarizer(classes=sorted(all_terms), sparse_output=True)
        y_train = mlb.fit_transform([protein_2_terms[p] for p in train_pids])
        y_val = mlb.transform([protein_2_terms[p] for p in val_pids])

        aspect_mlbs[aspect] = mlb
        num_classes = len(mlb.classes_)
        print(f"  GO terms: {num_classes}")

        ia_weights = torch.tensor([ia_dict.get(t, 1.0) for t in mlb.classes_], dtype=torch.float32)
        aspect_ia_weights[aspect] = ia_weights

        class_weights = compute_class_weights(y_train, num_classes, config.get('class_weight_smoothing', 0.1)) if config.get('use_class_weights') else None

        train_ds = ProtDatasetWithDomains(train_pids, y_train, embeddings_dict, pid_to_domain_multihot, NUM_DOMAINS)
        val_ds = ProtDatasetWithDomains(val_pids, y_val, embeddings_dict, pid_to_domain_multihot, NUM_DOMAINS)

        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size']*2, shuffle=False, num_workers=4, pin_memory=True)

        model = ProteinClassifierWithDomains(
            EMBED_DIM, num_classes, config['hidden_dims'], config['dropout'],
            num_domains=NUM_DOMAINS, domain_output_dim=CONFIG['domain_output_dim'], use_domains=USE_DOMAINS
        )
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        model, fmax, threshold = train_model(aspect, model, train_loader, val_loader, ia_weights, config, class_weights)

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

                # Store diagnostic info
                print(f"\n📊 Domain Impact Summary:")
                print(f"   Mean prediction change: {domain_impact:.4f} ({domain_impact*100:.2f}%)")
                print(f"   Predictions changed >1%: {ablation_result['percent_changed']:.1f}%")
                print(f"   Domain features {'ARE' if domain_used else 'ARE NOT'} being used effectively")

                if not domain_used:
                    print(f"\n⚠️  WARNING: Domains have minimal impact!")
                    print(f"   Consider:")
                    print(f"   1. Using ImprovedProteinClassifierWithDomains from domain_diagnostics.py")
                    print(f"   2. Increasing domain_output_dim to 512")
                    print(f"   3. See CHANGES_TO_MAIN_SCRIPT.md for fixes")
                else:
                    print(f"\n✅ Domains are contributing meaningfully to predictions!")

            except Exception as e:
                print(f"⚠️  Diagnostic test failed: {e}")

        # ===== END DIAGNOSTICS =====

        aspect_models[aspect], aspect_thresholds[aspect] = model, threshold
        results.append({
            'aspect': aspect,
            'name': name,
            'fmax': fmax,
            'threshold': threshold,
            'classes': num_classes,
            'domain_impact': domain_impact,
            'domain_used': domain_used
        })

        del train_loader, val_loader, train_ds, val_ds
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    for r in results:
        domain_status = ""
        if r['domain_impact'] is not None:
            impact_pct = r['domain_impact'] * 100
            status_icon = "✅" if r['domain_used'] else "⚠️ "
            domain_status = f" | Domains: {status_icon} {impact_pct:.2f}%"

        print(f"{r['aspect']:<4} {r['name']:<25} Fmax={r['fmax']:.4f} t={r['threshold']:.2f} classes={r['classes']}{domain_status}")

    avg_fmax = np.mean([r['fmax'] for r in results])
    print(f"\nAVG Fmax: {avg_fmax:.4f}")

    # Domain usage summary
    if any(r['domain_impact'] is not None for r in results):
        print(f"\n{'='*80}\nDOMAIN FEATURE USAGE SUMMARY\n{'='*80}")
        all_used = all(r['domain_used'] for r in results if r['domain_impact'] is not None)
        none_used = not any(r['domain_used'] for r in results if r['domain_impact'] is not None)

        if all_used:
            print("✅ All aspects are using domain features effectively!")
        elif none_used:
            print("⚠️  WARNING: Domain features are NOT being used in any aspect!")
            print("   See README_DOMAIN_SOLUTION.md for fixes")
        else:
            print("⚠️  MIXED: Some aspects use domains, others don't")
            print("   Review individual diagnostic outputs above")


# ===== PREDICTION =====
def predict(min_preds=5, max_preds=80):
    pid_test = [p for p in test_sequences if p in embeddings_dict]
    print(f"\nPredicting for {len(pid_test)} proteins")

    test_ds = TestDataSetWithDomains(pid_test, embeddings_dict, pid_to_domain_multihot, NUM_DOMAINS)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    results = []
    for aspect in ASPECTS:
        print(f"  {aspect}...")
        model, mlb, t = aspect_models[aspect], aspect_mlbs[aspect], aspect_thresholds[aspect]
        model.eval()
        batch_start = 0

        for X, domain in tqdm(test_loader, desc=aspect):
            X, domain = X.to(device), domain.to(device)
            bs = X.shape[0]

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = torch.sigmoid(model(X, domain_multihot=domain)).cpu().numpy()

            for i in range(bs):
                pid, scores = pid_test[batch_start + i], preds[i]
                above = np.where(scores >= t)[0]
                if len(above) < min_preds:
                    above = np.argsort(scores)[-min_preds:]
                if len(above) > max_preds:
                    above = above[np.argsort(scores[above])[-max_preds:]]

                for idx in above:
                    results.append({'pid': pid, 'term': mlb.classes_[idx], 'p': float(scores[idx])})
            batch_start += bs

    return pd.DataFrame(results)


def get_ancestors(term, go_parents, cache={}):
    if term in cache:
        return cache[term]
    ancestors = set()
    stack = [term]
    while stack:
        for parent in go_parents.get(stack.pop(), []):
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)
    cache[term] = ancestors
    return ancestors


def propagate(df, go_parents):
    print("Propagating...")
    rows = []
    for pid, group in tqdm(df.groupby('pid')):
        term_scores = {}
        for _, row in group.iterrows():
            term, score = row['term'], row['p']
            term_scores[term] = max(term_scores.get(term, 0), score)
            for anc in get_ancestors(term, go_parents):
                term_scores[anc] = max(term_scores.get(anc, 0), score)
        rows.extend({'pid': pid, 'term': t, 'p': s} for t, s in term_scores.items())
    return pd.DataFrame(rows)


def apply_negative_propagation(df, negative_keys):
    if not negative_keys:
        return df
    df['key'] = df['pid'].astype(str) + '_' + df['term'].astype(str)
    before = len(df)
    df = df[~df['key'].isin(negative_keys)].drop(columns=['key'])
    print(f"  Removed {before - len(df)} negative predictions")
    return df


def add_goa_ground_truth(df, goa_df):
    if goa_df is None or len(goa_df) == 0:
        return df
    test_pids = set(df['pid'].unique())
    goa_test = goa_df[goa_df['protein_id'].isin(test_pids)].rename(columns={'protein_id': 'pid', 'go_term': 'term', 'score': 'p'})[['pid', 'term', 'p']]
    return pd.concat([df, goa_test]).groupby(['pid', 'term'])['p'].max().reset_index()


# ===== RUN =====
if __name__ == "__main__":
    train_all()

    print(f"\n{'='*80}\nGENERATING PREDICTIONS\n{'='*80}")
    submission = predict(min_preds=5, max_preds=80)
    submission = propagate(submission, go_parents)
    submission = apply_negative_propagation(submission, negative_keys)
    submission = add_goa_ground_truth(submission, goa_positive_df)

    submission = submission.sort_values(['pid', 'p'], ascending=[True, False])
    submission = submission.groupby('pid').head(1500)
    submission[['pid', 'term', 'p']].to_csv('submission.tsv', sep='\t', index=False, header=False)

    print(f"\n{'='*80}\nDONE\n{'='*80}")
    print(f"Predictions: {len(submission):,}, Proteins: {submission['pid'].nunique():,}")
    print(f"Model: V10 (ESM-650M + InterPro/Signature domains)")
