"""
train_model.py — Train a LightGBM binary classifier on the 3M reference vectors.
Exports a compact binary model readable by GbmPredictor.cs.

Binary format:
  [n_trees: int32]
  For each tree:
    [n_nodes: int32]
    For each node (13 bytes):
      [feature_index: uint8]  (0xFF = leaf)
      [threshold: float32]
      [left_child: int16]
      [right_child: int16]
      [leaf_value: float32]
"""

import gzip
import json
import struct
import sys
import numpy as np
import lightgbm as lgb
from pathlib import Path


def load_references(path: str):
    """Load reference vectors from JSON (optionally gzipped)."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    n = len(data)
    X = np.empty((n, 14), dtype=np.float32)
    y = np.empty(n, dtype=np.int32)

    for i, entry in enumerate(data):
        X[i] = entry['vector']
        y[i] = 1 if entry['label'] == 'fraud' else 0

    return X, y


def train_model(X, y):
    """Train LightGBM classifier."""
    train_data = lgb.Dataset(X, label=y, free_raw_data=False)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 200,
        'max_depth': 10,
        'learning_rate': 0.05,
        'min_child_samples': 30,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'reg_lambda': 1.0,
        'reg_alpha': 0.5,
        'verbose': -1,
        'num_threads': -1,
        'seed': 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=350,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(50)],
    )

    return model


def export_model_binary(model, output_path: str):
    """Export LightGBM model to compact binary format."""
    model_dump = model.dump_model()
    trees = model_dump['tree_info']

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<i', len(trees)))

        for tree_info in trees:
            tree = tree_info['tree_structure']
            nodes = []
            _flatten_tree(tree, nodes)

            f.write(struct.pack('<i', len(nodes)))
            for node in nodes:
                f.write(struct.pack(
                    '<Bfhhf',
                    node['feature'],
                    node['threshold'],
                    node['left'],
                    node['right'],
                    node['leaf_value'],
                ))

    print(f"Exported {len(trees)} trees to {output_path}")


def _flatten_tree(node, nodes, depth=0):
    """Recursively flatten a LightGBM tree into a node array."""
    idx = len(nodes)

    if 'leaf_value' in node and 'split_feature' not in node:
        # Leaf node
        nodes.append({
            'feature': 0xFF,
            'threshold': 0.0,
            'left': 0,
            'right': 0,
            'leaf_value': node['leaf_value'],
        })
        return idx

    # Internal node
    nodes.append({
        'feature': node['split_feature'],
        'threshold': node['threshold'],
        'left': 0,
        'right': 0,
        'leaf_value': 0.0,
    })

    left_idx = _flatten_tree(node['left_child'], nodes, depth + 1)
    nodes[idx]['left'] = left_idx

    right_idx = _flatten_tree(node['right_child'], nodes, depth + 1)
    nodes[idx]['right'] = right_idx

    return idx


def evaluate_model(model, X, y):
    """Print accuracy stats."""
    probs = model.predict(X)

    # At confidence thresholds
    for low, high in [(0.20, 0.80), (0.15, 0.85), (0.10, 0.90), (0.25, 0.75)]:
        confident_mask = (probs < low) | (probs > high)
        confident_pct = confident_mask.mean() * 100
        if confident_mask.sum() > 0:
            pred_labels = (probs[confident_mask] >= 0.5).astype(int)
            acc = (pred_labels == y[confident_mask]).mean() * 100
            print(f"  Threshold ({low}/{high}): {confident_pct:.1f}% confident, {acc:.2f}% accurate")

    # Overall accuracy
    pred_labels = (probs >= 0.5).astype(int)
    overall_acc = (pred_labels == y).mean() * 100
    print(f"  Overall accuracy: {overall_acc:.2f}%")

    # Distribution analysis
    ambiguous_mask = (probs >= 0.20) & (probs <= 0.80)
    ambiguous_pct = ambiguous_mask.mean() * 100
    print(f"  Ambiguous (0.20-0.80): {ambiguous_pct:.1f}% of requests → IVF fallback")


def main():
    refs_path = sys.argv[1] if len(sys.argv) > 1 else '/data/references.json.gz'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '/data/model.bin'

    print(f"Loading references from {refs_path}...")
    X, y = load_references(refs_path)
    print(f"Loaded {len(X)} vectors ({y.sum()} fraud, {len(y) - y.sum()} legit)")

    print("Training LightGBM model...")
    model = train_model(X, y)

    print("\nModel evaluation:")
    evaluate_model(model, X, y)

    print(f"\nExporting to {output_path}...")
    export_model_binary(model, output_path)
    print("Done!")


if __name__ == '__main__':
    main()
