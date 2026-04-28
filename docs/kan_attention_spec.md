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

---

## 5. Core Components

The KAN attention normaliser is composed of five interacting components. Each is described in its own subsection.

### 5.1 B-Spline Normaliser (per head)

**Parametrisation.**

- Each head owns one B-spline `φ : ℝ → ℝ` defined on a fixed grid of knot positions covering the expected range of the attention scores `s_ij = (Q · Kᵀ)_{ij} / √d`.
- The spline is `φ(s) = Σ_{j=1}^{N} c_j · B_j(s)`, where:
  - `B_j(·)` are the (immutable) B-spline basis functions of order `k`.
  - `c_j ∈ ℝ` are the learnable coefficients (the entire mutable state of the head's normaliser).
  - `N` is the number of knots (basis count).

**Locality.** B-spline basis functions of order `k` have **local support over `k+1` adjacent knot intervals**. Consequently each coefficient `c_j` only influences `φ(s)` for `s` in a small neighbourhood around its knot. This locality is load-bearing for the entire design:

- It defines the natural **window** for mechanism #1 (§5.2): the window of `c_i` is the set of coefficients whose support overlaps `c_i`'s support, naturally `2k+1` coefficients on a uniform grid.
- It allows the autonomous local learning rule (§6.4) to update only the small subset of coefficients whose basis functions are non-zero at the observed score, keeping per-update cost O(`k+1`) rather than O(`N`).

**Output and normalisation.**

- The forward attention weights are computed by applying `φ` pointwise to every score in the row-major attention matrix and then row-normalising:
  ```
  w_ij = φ(s_ij) / Σ_j' φ(s_ij')
  ```
- Because the output is row-normalised, **multiplying every coefficient `c_j` by any positive constant leaves `w_ij` unchanged**. This scale-invariance is the gauge freedom exploited by the GM=1 fixing (§5.3).
- For the output to constitute a valid attention distribution it must be non-negative. A non-negativity constraint on `φ(s)` is required (see §5.1 Constraints below).

**Constraints.**

| Constraint | Reason | Enforcement |
|---|---|---|
| `φ(s) ≥ 0` for all `s` in the grid range | Attention weights must be non-negative | Either (a) parametrise the spline as `φ(s) = ψ(s)²` with an underlying spline `ψ`, or (b) accept occasional negative outputs and clip to zero with a small floor `ε_φ`. **Default: option (a).** |
| Coefficients sign-preserving across rebalance | Stability of the spline shape under mechanism #1 | Mechanism #1's multiplicative redistribution always uses positive factors. |
| GM=1 per head | Canonical gauge; meaningful thresholds | Renormalisation step in the per-inference pipeline (§7). |

**Initialisation.**

- At layer construction (or first retrofit load), coefficients are fitted such that `φ(s) ≈ exp(s)` over the grid range. A simple least-squares fit on a dense grid of `(s, exp(s))` pairs suffices.
- Phase M therefore starts with the KAN already approximating softmax. `KL_ema` begins near zero rather than at random, dramatically shortening Phase M for retrofit deployments.

**Grid configuration.**

- Grid range: covers the expected `Q·Kᵀ/√d` range. Default `[−8, +8]` nats; tunable per layer if scores are known to live in a different range.
- Knot count `N`: trades expressivity for memory. Default `N = 64`. Each head holds `N` floats of state.
- Basis order `k`: cubic (`k = 3`) by default, giving window size `2k+1 = 7`.
- Grid is **immutable** for the lifetime of a layer. (A v3 extension may add online grid refinement; not in v1.)

**Memory footprint per head.**

- `N` floats for coefficients.
- One `N`-sized scratch buffer for evaluation (shared across heads at runtime if memory is tight).
- Negligible compared to Q/K/V projection weights, which dominate per-layer memory.

### 5.2 Mechanism #1 — GM-Based Coefficient Rebalance

The central regulator of the spline. Runs every inference step on every head, after the local learning rule (§6.4) has applied its update. Iterates to fixed point with a hard safety cap on iterations.

#### 5.2.1 Window

For each coefficient `c_i`, its **window** is the set of coefficients whose B-spline supports overlap `c_i`'s support. On a uniform grid with order-`k` basis this is the `2k+1` coefficients centred on `c_i` (clamped near boundaries). Groups self-form from the basis structure; no manual group definition is required.

The window is the unit on which all of mechanism #1's invariants are defined.

#### 5.2.2 Window Statistic — Geometric Mean

For each window, compute:
```
g = exp( mean_{j ∈ window}( log |c_j| ) )
```

Computed in log-space for numerical safety. No `+ε` floor is required because the low-lift rule (§5.2.4) keeps `|c_j|` strictly bounded away from zero.

The geometric mean is the right statistic for a multiplicative regime: it is scale-equivariant (`GM(α·X) = α·GM(X)`), it is naturally robust to a single dominating outlier (the GM is barely affected by extreme single values, unlike the arithmetic mean), and it is the correct centre of mass under the GM=1 gauge (§5.3).

#### 5.2.3 High Clip

If `|c_i| > 3 · g`:

1. Clip the offending coefficient: `c_i ← sign(c_i) · 3 · g`.
2. Compute the multiplicative excess: `r = |c_i_old| / |c_i_new| > 1`.
3. Redistribute `r` across the remaining `N−1 = 2k` window coefficients by multiplicative factors `f_j` satisfying:
   ```
   c_j ← c_j · f_j        for j ≠ i
   Π_{j ≠ i} f_j  =  r
   ```
   The factors are positive (sign-preserving). The exact distribution rule depends on the share mode (§5.2.5).

By construction, the **window product `Π_{j ∈ window} |c_j|` is preserved exactly** by this operation:
```
Π_new = (|c_i| / r) · (Π_{j ≠ i} |c_j| · f_j)
      = (|c_i| / r) · r · Π_{j ≠ i} |c_j|
      = Π_old
```

#### 5.2.4 Low Lift (Productivity Rule)

If `|c_i| < g / 2`:

1. Lift the deficient coefficient toward the window's GM: `c_i ← sign(c_i) · g`.
2. Compute the multiplicative deficit: `q = |c_i_new| / |c_i_old| > 1`.
3. Shrink the remaining window coefficients by multiplicative factors `f_j` satisfying:
   ```
   c_j ← c_j · f_j        for j ≠ i
   Π_{j ≠ i} f_j  =  1 / q
   ```

Same product-preservation property as the high clip.

**Threshold rationale.** The low threshold is `g / 2`, not the symmetric `g / 3`. The asymmetry is intentional: the low-lift rule is doing **productivity** work (no dead weight), not merely numerical safety. Keeping every coefficient contributing maximises the per-head representational budget, which makes the head squaring (§5.5) trigger a meaningful signal of genuine capacity demand rather than wasted-budget noise. See §5.5 for the coupling.

#### 5.2.5 Share Rule for Redistribution

The factors `f_j` are computed as `f_j = r^{s_j}` (high clip) or `f_j = (1/q)^{s_j}` (low lift), where the **share weights `s_j` sum to 1** over the redistributed coefficients.

Two share rules are supported, controlled per-event:

- **Proportional:** `s_j ∝ |c_j| / Σ_{k ≠ i} |c_k|`. Larger neighbours absorb more growth (high-clip) or shrinkage (low-lift).
- **InverseProportional:** `s_j ∝ (1 / |c_j|) / Σ_{k ≠ i} (1 / |c_k|)`. Smaller neighbours absorb more.

Two flags select rules independently:
```
HighClipShare ∈ { Proportional, InverseProportional, Auto }
LowLiftShare  ∈ { Proportional, InverseProportional, Auto }
```

##### Mode taxonomy and stability

| Combination | Name | High-clip behaviour | Low-lift behaviour | Stability |
|---|---|---|---|---|
| Prop, Prop | Mode A — *preserve shape* | Big neighbours grow more (creates outliers) | Big neighbours shrink more (flattens) | Mixed: bad on high-clip, good on low-lift |
| Inv, Inv | Mode B — *reverse* | Small neighbours grow more (flattens) | Small neighbours shrink more (creates new zeros) | Mixed: good on high-clip, bad on low-lift |
| Inv, Prop | **Mode C — *always-flatten*** | Flattens (good half of B) | Flattens (good half of A) | **Stable; converges fast** |
| Prop, Inv | Mode D — *anti-flatten* | Creates outliers + creates zeros | — | **Pathological; oscillates. Excluded.** |

Mode C isolates the stabilising half of each pure mode and is strictly the safest. Mode A is useful when sharpness preservation is desired and clip pressure is low. Mode B is rarely useful in isolation. Mode D must never be selected.

##### Auto mode (default for both flags)

`Auto` selects the mode based on the layer's current `clip_rate` (defined in §5.5):

```
if clip_rate < τ_calm:                          -- e.g. 2 %
    HighClipShare := Proportional               -- Mode A: preserve shape
    LowLiftShare  := Proportional
elif clip_rate > τ_stressed:                    -- e.g. 10 %
    HighClipShare := InverseProportional        -- Mode C: always-flatten
    LowLiftShare  := Proportional
else:
    -- hysteresis band: keep current mode
```

Mode D is structurally excluded by the Auto logic.

#### 5.2.6 Cascade to Fixed Point

After processing all coefficients in one sweep, a coefficient that was a *recipient* of redistribution may now itself violate the high-clip or low-lift threshold (because its window's `g` has shifted). Mechanism #1 therefore iterates:

```
for iter in 1 .. max_iter:
    fired = 0
    for i in 1 .. N:
        recompute g for c_i's window
        if |c_i| > 3·g: high-clip; fired += 1
        elif |c_i| < g/2: low-lift; fired += 1
    if fired == 0: break       -- fixed point reached
```

The cascade is guaranteed to reduce total per-window deviation from the GM in the long-run average (the share rule choice in Mode C makes each step variance-reducing). The hard cap `max_iter = 16` is a safety guard against pathological inputs; in practice the cascade typically converges in 1–3 iterations.

#### 5.2.7 Counters Updated

Each invocation of mechanism #1 updates two counters used elsewhere in the spec:

- `clip_rate` (per layer): exponential moving average of the fraction of coefficients firing clip-or-lift in the most recent sweep. Used by Auto mode (§5.2.5) and by head squaring (§5.5).
- `clip_count_total` (per head, per phase): cumulative count of clips and lifts since checkpoint load. Diagnostic only.

#### 5.2.8 Invariants Maintained

After each invocation of mechanism #1, on every head:

- **(M1-1)** Per-window product: `Π_{j ∈ window} |c_j|` is unchanged from before the sweep, up to floating-point tolerance.
- **(M1-2)** Sign of every coefficient is unchanged.
- **(M1-3)** No coefficient is below `g/2` or above `3·g` of its own window's `g`, after cascade termination.
- **(M1-4)** Scale-equivariance: applying mechanism #1 to coefficients `(c_j)` then multiplying by `α > 0` produces the same result as multiplying by `α` first then applying mechanism #1.





