# Adapting Tabular Foundation Models Adaptation for Survival Analysis


[![Paper](https://img.shields.io/badge/paper-under%20review-yellow)](.)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

## Abstract

Tabular foundation models (TabFMs) have emerged as a powerful paradigm for structured data, yet their transfer to survival analysis—where censoring, temporal structure, and competing risks create objective mismatch—remains unclear. We conduct the first comprehensive benchmark of foundation-model-based approaches for survival analysis, evaluating multiple methods across a collection of single-risk and competing-risk datasets with three categories of adaptation strategies: zero-shot in-context inference, classification-based head fine-tuning, and survival-aware objectives. Our results prove the effectiveness of adaptation strategies for survival analysis tasks: supervised fine-tuning substantially outperforms zero-shot inference, and survival-specific heads (Cox, MTLR, DeepHit) consistently outperform classification-based alternatives. In single-risk settings, Cox-based fine-tuning dominates both discrimination and calibration metrics, while in competing-risk settings, MTLR achieves top performance through explicit probability conservation. Meanwhile, classification-based approaches universally fail on calibration metrics, revealing fundamental incompatibility between independent binary objectives and survival's distributional constraints. These findings establish foundation models as a promising paradigm for survival analysis with appropriate adaptation.

---
