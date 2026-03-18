"""
Diversity analysis for generated text data.
Computes distinct-1/2/3, self-BLEU, average length, and label distribution.

Usage:
    python -m task.classification.analyze_diversity --input_path <path_to_pickle>
"""
import os
import sys
import pickle
import argparse
import random
from collections import Counter

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def load_generated_data(pkl_path: str) -> dict:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_distinct_n(texts: list, n: int) -> float:
    """Compute distinct-n: ratio of unique n-grams to total n-grams."""
    total_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        total_ngrams.extend(ngrams)
    if len(total_ngrams) == 0:
        return 0.0
    return len(set(total_ngrams)) / len(total_ngrams)


def compute_self_bleu(texts: list, sample_size: int = 1000) -> float:
    """Compute self-BLEU: average BLEU of each sentence against all others (sampled)."""
    if len(texts) < 2:
        return 0.0
    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)

    smoothing = SmoothingFunction().method1
    scores = []
    for i in range(len(texts)):
        hypothesis = texts[i].lower().split()
        references = [texts[j].lower().split() for j in range(len(texts)) if j != i]
        # Sample references to keep computation manageable
        if len(references) > 100:
            references = random.sample(references, 100)
        score = sentence_bleu(references, hypothesis, smoothing_function=smoothing)
        scores.append(score)
    return np.mean(scores)


def compute_avg_length(texts: list) -> float:
    lengths = [len(text.split()) for text in texts]
    return np.mean(lengths) if lengths else 0.0


def compute_label_distribution(labels: list, num_classes: int) -> dict:
    counter = Counter(labels)
    dist = {}
    for i in range(num_classes):
        dist[i] = counter.get(i, 0)
    return dist


def analyze(pkl_path: str):
    print(f"Analyzing: {pkl_path}")
    data = load_generated_data(pkl_path)

    texts = data['input_text']
    labels = data['labels']
    num_classes = data.get('num_classes', max(labels) + 1 if labels else 2)

    print(f"\nTotal samples: {len(texts)}")
    print(f"Num classes: {num_classes}")

    # Average length
    avg_len = compute_avg_length(texts)
    print(f"Average length (words): {avg_len:.2f}")

    # Distinct-n
    for n in [1, 2, 3]:
        dn = compute_distinct_n(texts, n)
        print(f"Distinct-{n}: {dn:.4f}")

    # Self-BLEU (sampled)
    print("Computing self-BLEU (this may take a moment)...")
    sbleu = compute_self_bleu(texts, sample_size=1000)
    print(f"Self-BLEU (sampled): {sbleu:.4f}")

    # Label distribution
    label_dist = compute_label_distribution(labels, num_classes)
    print(f"\nLabel distribution:")
    for label_idx, count in label_dist.items():
        pct = count / len(labels) * 100 if labels else 0
        print(f"  Label {label_idx}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze diversity of generated data')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the pickle file')
    args = parser.parse_args()
    analyze(args.input_path)
