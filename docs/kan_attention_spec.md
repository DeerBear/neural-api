# KAN Attention Normaliser — Design Specification

**Version:** v1 (mechanism-locked, pre-implementation)
**Scope:** language-independent design document

---

## 1. Abstract

A drop-in replacement for the softmax operator inside multi-head self-attention. The layer is trained entirely as standard softmax attention; at inference time a per-head B-spline normaliser (a "KAN edge") autonomously learns to mimic softmax, takes over the forward path per-layer when converged, and thereafter evolves to surface information that a fixed-shape softmax cannot express. Retrofits onto any pre-trained softmax transformer with no retraining.

**Non-goals:** KAN-ifying Q, K, V or output projections; replacing any part of training; supporting training-time gradient flow through the KAN normaliser.

---

## 2. Design Goals

The design is a layered response to a single overarching goal: **make more information available at inference than a fixed-shape softmax can express, without retraining and without compromising deployment safety.** Each mechanism in the spec exists to discharge a specific sub-goal:

| Goal | Mechanism delivering it |
|---|---|
| Sharper attention on fewer heads | B-spline shape freedom beats fixed `exp` |
| Safe deployment of novel machinery | Shadow-mimic-then-takeover; softmax is the known baseline |
| No gradient machinery at inference | Normalised LMS (autonomous local rule, no autograd) |
| Surface information a small model would lose | Post-takeover self-distillation + product-preserving budget |
| Self-regulating capacity | Head squaring with self-limiting dynamics |
| Full reproducibility | Deterministic-given-state; all state checkpointable |
| Retrofit on existing models | Training is untouched; any pre-trained softmax model plugs in |

**Explicit non-goals:**

- Replacing softmax during training. Training stays standard.
- Replacing Q/K/V or output projections in v1. They contribute no nonlinearity, and they already receive full task-loss gradient during training; KAN-ifying them would be cargo-culting.
- Supporting backpropagation through the KAN at inference. All inference-time evolution is driven by autonomous local rules.
- Maintaining backwards-bit-compatibility with the original softmax forward path after a layer hands over. The whole point is that the KAN diverges from softmax post-takeover.

