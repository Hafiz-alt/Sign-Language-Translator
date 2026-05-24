"""Compatibility entry point for real model evaluation.

This file previously generated invented accuracy values. It now runs the
actual held-out confusion matrix computation.
"""

from plot_confusion_matrix import generate_confusion_matrix


if __name__ == "__main__":
    generate_confusion_matrix()
