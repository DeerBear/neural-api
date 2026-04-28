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

---

## 3. Training Phase

Training is **completely vanilla softmax attention**. Q, K, V and output projections remain standard linear layers. The softmax operator is the standard softmax. **No KAN state exists during training.**

Consequences of this choice:

- **Indistinguishable trained model.** A model trained against this specification's attention layer is bit-identical (up to RNG seeding) to one trained against the host library's existing softmax attention. Standard checkpoints work.
- **No new gradient paths.** Backpropagation flows through the standard softmax exactly as it always has. No new gradient machinery, no second optimiser, no auxiliary loss.
- **No new hyperparameters during training.** Learning rate schedules, batch sizes, regularisation, etc. are unchanged.
- **Retrofit is trivial.** Any pre-trained softmax transformer (regardless of when, where or how it was trained) can be loaded into this layer. The KAN state is initialised at first forward pass, not at training time.
- **Failure mode is graceful.** If the KAN machinery is disabled at inference (for whatever reason), the layer is functionally identical to standard softmax attention. There is no path by which "training broke because the KAN was being trained."

**Implementation note (abstract).** The new layer class must, in its training-time forward and backward implementation, be byte-identical in behaviour to the host library's existing softmax attention layer. The KAN-specific state and operations are gated by an `inference-mode` flag that is `false` during training and only flips `true` once the network enters inference. This gating is the single point of departure from the existing layer.

---

## 4. Inference Lifecycle

At inference time, each head of each attention layer progresses independently through three phases. Layers do not have to be in the same phase as one another; per-head state is independent.

### 4.1 Phase M — Mimicry (default on fresh deploy)

- Forward path uses **softmax**.
- A per-head B-spline `φ(s) = Σ_j c_j · B_j(s)` runs in **shadow** alongside the softmax. It receives every score the softmax sees but its output does not affect the layer's attention weights.
- An **autonomous local rule** updates the spline coefficients toward the target `exp(s)` (see §6.4 for the rule, §5.1 for the spline parametrisation).
- A convergence metric `KL(softmax_weights ‖ KAN_weights)` is tracked as an exponential moving average (`KL_ema`).
- The phase is observation-only from the model's behaviour standpoint: the deployed model's attention outputs are exactly those of the standard softmax model. **There is zero behavioural risk during this phase.**

### 4.2 Phase T — Takeover (atomic transition)

- Triggered when `KL_ema < ε_KL` for at least `N_confirm` consecutive forward passes (see §6 for criteria and defaults).
- Performed atomically per head per layer:
  1. The forward path switches from `softmax(s)` to `φ(s) / Σφ(s)`.
  2. The Phase-M mimicry rule retires.
  3. The Phase-D divergence rule activates.
  4. The per-head `status` flag flips from `SoftmaxActive` to `KANActive`.
- **One-way.** No mechanism reverts a head from `KANActive` back to `SoftmaxActive`. Stability of the post-takeover dynamics is the responsibility of mechanism #1 (§5.2) and head squaring (§5.5).
- **Per-head independence.** Different heads of the same layer may take over at different times. Different layers progress entirely independently.

### 4.3 Phase D — Divergence (post-takeover)

- Forward path uses the **KAN**.
- The local rule is no longer mimicry; it is **self-distillation toward a sharpened version of the head's own current attention distribution** (§6.4).
- Combined with mechanism #1's product-preserving redistribution (§5.2), the sharpening rule cannot grow mass — it can only *redistribute* mass within the head's GM=1 budget. This is the mechanism by which information that softmax would have averaged-out gets concentrated into peaks at score regions that consistently carry signal.
- Phase D runs **indefinitely**. There is no terminal state; the head keeps adapting to the inference data distribution as long as inference continues.

### 4.4 Lifecycle State Diagram

```
                          per head, per layer

   ┌──────────────────┐      KL_ema < ε_KL       ┌──────────────────┐
   │                  │      sustained over      │                  │
   │     Phase M      │  ──── N_confirm ───▶     │     Phase D      │
   │     (Mimicry)    │       inferences         │   (Divergence)   │
   │                  │                          │                  │
   │ status =         │                          │ status =         │
   │   SoftmaxActive  │                          │   KANActive      │
   │ forward = softmax│                          │ forward = KAN    │
   │ rule = NLMS→exp  │                          │ rule = self-dist │
   │                  │                          │                  │
   └──────────────────┘                          └──────────────────┘
                                                         │
                                                         │ no transition
                                                         ▼
                                                    (terminal)
```

### 4.5 Non-Convergent Heads

If a head's `KL_ema` never satisfies the convergence criterion (e.g. score distribution is non-stationary or the spline grid is misconfigured for the observed range), the head simply remains in Phase M indefinitely. The layer continues to serve correctly via softmax. This is graceful degradation, not failure.

A head that remains in Phase M at checkpoint time is saved with `status = SoftmaxActive`; on reload, it resumes mimicry from its current coefficient state.



