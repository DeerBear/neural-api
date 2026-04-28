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

### 5.3 GM=1 Gauge Fixing (per head)

#### 5.3.1 Premise — The Gauge Freedom

Because the attention weights are computed by row-normalisation:
```
w_ij = φ(s_ij) / Σ_j' φ(s_ij')
```
multiplying every coefficient `c_j` by any positive scalar `α > 0` leaves `w_ij` **identically unchanged**. The KAN therefore has a **gauge freedom**: a one-parameter family of coefficient configurations that all produce the same forward output.

This freedom is the only direction in which the coefficient vector can drift without changing the model's behaviour. Pinning that direction is free — it costs the model nothing in expressive power — and pays off in stability, interpretability, and the meaning of mechanism #1's thresholds.

#### 5.3.2 The Gauge Choice — `GM = 1`

The chosen gauge is **per-head geometric mean equal to 1**:
```
G(head) := exp( mean_j( log |c_j| ) )  =  1
```

Equivalently: the sum of log-magnitudes of all coefficients in a head is zero. Some coefficients have `|c_j| < 1`, some have `|c_j| > 1`, and they multiplicatively balance.

**Why per-head, not per-layer:** each head is an independent representational budget (§5.5). Per-head gauge gives each head its own canonical scale; per-layer gauge would tie heads together and dilute the "more heads = more budget" interpretation.

#### 5.3.3 The Renormalisation Operation

After mechanism #1 has converged in a given inference step (§5.2.6), per head:

```
G   := exp( mean_j( log |c_j| ) )         -- in log-space, O(N)
c_j ← c_j / G                              -- uniform rescale, O(N)
```

That is the entire operation. One pass to compute `G`, one pass to apply.

**Numerical implementation note.** Use log-space throughout: maintain or recompute `Σ_j log|c_j|`, divide by `N` to get `log G`, then `c_j ← c_j · exp(−log G)`. This avoids the `Π|c_j|` underflow/overflow that direct GM computation would suffer for `N` in the dozens.

#### 5.3.4 Why It Costs Nothing in Forward Output

Before renormalisation, the spline outputs `φ(s) = Σ_j c_j · B_j(s)`. After renormalisation by uniform factor `1/G`:
```
φ_new(s) = Σ_j (c_j / G) · B_j(s) = (1/G) · Σ_j c_j · B_j(s) = φ_old(s) / G
```
And the row-normalised attention weight:
```
w_new = φ_new(s) / Σ φ_new = (φ_old(s)/G) / (Σ φ_old/G) = φ_old(s) / Σ φ_old = w_old
```

The attention output is **bit-identical** before and after renormalisation (up to floating-point reordering of operations). The renormalisation is a pure gauge transformation.

#### 5.3.5 Composition with Mechanism #1

Mechanism #1 preserves **per-window products** (M1-1). The GM=1 renormalisation applies a uniform multiplicative rescaling, which **changes** every window product by `(1/G)^(2k+1)`. There is no contradiction: mechanism #1 preserves whatever window products it finds at the start of its sweep. The renormalisation simply resets the "preserved targets" to the new gauge.

The locked ordering (also encoded in §7) is:

```
1. local learning update (additive, perturbs both window products and G)
2. mechanism #1 sweep (cascades to fixed point; preserves whatever products it sees)
3. GM = 1 renormalisation (uniform rescale; resets gauge)
```

After step 3, the head is in a canonical state: GM=1 and all window products consistent with that gauge.

#### 5.3.6 What the Gauge Buys

- **Meaningful thresholds.** Mechanism #1's `3·g` and `g/2` thresholds are now expressed against a stable reference. With GM=1, the typical window `g` is approximately 1, so coefficients live (approximately) in `[1/6, 6]`. Thresholds remain comparable across layers, across heads, and across time.
- **No coefficient-magnitude drift.** Without the gauge, the additive learning updates would slowly inflate or deflate the overall coefficient scale (depending on the data distribution). The gauge eliminates this nuisance dynamic entirely.
- **Numerical stability.** `log |c_j|` is bounded; `Σ log |c_j|` cannot underflow or overflow; the GM computation is well-conditioned regardless of `N`.
- **Foundation for head squaring.** With the gauge in place, each head's representational budget is well-defined (the bounded coefficient interval), making "this head has saturated its budget" a meaningful statement (§5.5).

#### 5.3.7 Invariants Maintained

After step 7 of the per-inference pipeline (§7), on every head:

- **(GM-1)** `|G − 1| < ε_gauge` where `ε_gauge` is at machine-precision tolerance for the head's coefficient count.
- **(GM-2)** The forward attention weights computed before and after renormalisation are bit-identical (up to FP reordering).
- **(GM-3)** Every coefficient sign is preserved.

### 5.4 Multi-Head Dynamics — Head Squaring

#### 5.4.1 Premise — Each Head Is a Bounded Budget

After mechanism #1 + GM=1 gauge, each head's coefficients live in a bounded interval (approximately `[1/6, 6]` in absolute value, depending on window size). The spline `φ` constructed from those coefficients can express only patterns that fit within this **multiplicatively balanced budget**:

- Cannot have all coefficients large.
- Cannot have all coefficients tiny.
- Must distribute "strength" unevenly with geometric balance around 1.

If the data calls for an attention pattern that cannot fit within one head's budget — for example, multiple sharp peaks across distinct score regions, each requiring high-magnitude coefficients — the head will repeatedly bump against mechanism #1's clip thresholds. **Head squaring is the relief valve: when one head's budget saturates, allocate more heads.**

#### 5.4.2 Allocation — Pre-Allocate, Mask the Rest

At layer construction:

```
H_max := ⌊ embedding_dim / 2 ⌋    -- architectural ceiling: minimum 2 channels per head
ActiveHeads := 2                  -- specification-mandated initial value
```

All `H_max` heads are **physically allocated** at construction time (coefficients, RNG state, counters). Heads with index `≥ ActiveHeads` are **inactive**:

- Their KAN coefficients are not updated.
- Their forward contribution is **masked to zero** in the multi-head concat step.
- They consume memory but no meaningful compute.

Pre-allocation eliminates runtime tensor resize, structural network mutation, and the entire class of bugs that come with growing data structures during inference. Squaring is implemented as a single integer assignment plus the per-head copy/perturb of §5.4.4.

#### 5.4.3 Squaring Trigger

Per layer, two counters drive the trigger:

- `clip_rate_ema`: per-layer EMA of the fraction of all coefficients (across all *active* heads) that fired clip-or-lift in the most recent mechanism-#1 sweep.
- `consecutive_high_sweeps`: count of consecutive sweeps where `clip_rate_ema > τ_squaring`.

Trigger condition:

```
if ActiveHeads < H_max
   and consecutive_high_sweeps ≥ K_squaring:
       fire_squaring()
       consecutive_high_sweeps := 0
```

Defaults: `τ_squaring = 0.10` (10 %), `K_squaring = 64` consecutive sweeps. The latter prevents single-batch noise from triggering squaring.

#### 5.4.4 The Squaring Operation

```
H_old := ActiveHeads
H_new := min(H_old · H_old, H_max)        -- square, clamp at ceiling
spawn_per_parent := H_new / H_old          -- integer

for parent_idx in 0 .. H_old - 1:
    for child_local in 1 .. spawn_per_parent - 1:    -- skip 0 (parent keeps slot)
        child_idx := H_old + parent_idx · (spawn_per_parent - 1) + (child_local - 1)
        copy parent's B-spline coefficients into child slot
        for each c_j in child's coefficients:
            c_j ← c_j · exp( N(0, σ²) )    -- log-normal perturbation, seeded RNG
        copy parent's clip counters / KL EMA into child as starting state

ActiveHeads := H_new
```

Properties:

- Each existing head retains its slot and its coefficients unchanged.
- `H_new − H_old` new heads are populated as perturbed clones of the existing heads.
- The perturbation is **multiplicative log-normal** with `σ ≈ 0.01`. Multiplicative is consistent with the GM=1 gauge: it preserves expected `log |c_j|` (so the new head also satisfies GM=1 to first order) and breaks symmetry without large jumps.
- The RNG used is the **per-layer seeded RNG**, so the operation is deterministic given the saved state (§9).
- Because GM=1 is approximately preserved by the perturbation, the new heads' coefficients land in the same canonical range; no re-gauge is needed at squaring time (the next per-inference step's gauge fixing will tidy any first-order drift).

#### 5.4.5 Growth Pattern and Self-Limitation

Squaring produces **super-exponential capacity growth**:

```
ActiveHeads:  2 → 4 → 16 → 256 → 65536 → ...   (clamped at H_max)
```

Per-head clip pressure drops **roughly linearly in head count** (each head specialises to a smaller fraction of the input distribution). The asymmetry between super-exponential capacity growth and linear pressure relief produces **inherent self-limitation**:

- One squaring event typically reduces `clip_rate_ema` by a factor of `H_new / H_old`.
- After one or two events, `clip_rate_ema < τ_squaring`, the trigger goes silent, and `ActiveHeads` freezes.

**No adaptive threshold is required.** The dynamics produce stable equilibrium on their own. Different layers settle at different `ActiveHeads` values based on their data complexity.

#### 5.4.6 Coupling with the Productivity Low-Lift

The aggressive low-lift threshold (`g/2`, §5.2.4) is essential to the meaning of the squaring trigger:

- **Without aggressive low-lift:** some coefficients drift toward zero and become dead weight. A head's effective capacity is `M < N`, and the active `M` coefficients overload, raising `clip_rate` for reasons that have nothing to do with genuine capacity demand. Squaring fires prematurely on wasted-budget noise.
- **With aggressive low-lift:** every coefficient stays productive. Effective capacity equals nominal capacity. `clip_rate` rises only when the head genuinely needs more representational room. Squaring fires only when capacity expansion is actually warranted.

The two mechanisms together promote head squaring from "noisy event" to "meaningful diagnostic signal."

#### 5.4.7 One-Way Only — No Head Merging

Heads never merge. `ActiveHeads` is monotonic non-decreasing across a deployment run.

Rationale: merging two heads' splines requires choosing how to combine their coefficients. Any combination (average, weighted average, etc.) destroys learned structure that the two heads had specialised for. Mechanism #1's low-lift already handles coefficient-level rescue; head-level merging would risk catastrophic forgetting for no clear gain.

If a deployment scenario requires reducing `ActiveHeads` (e.g. reducing serving cost), the only supported approach is **load a checkpoint from before the squaring event** — not in-place merge.

#### 5.4.8 Failure Mode — Hitting the Ceiling

If `clip_rate_ema` remains above `τ_squaring` even with `ActiveHeads = H_max`, squaring is impossible. The layer enters a **degraded-but-stable regime**:

- Mechanism #1 fires more often than typical (high clip rate).
- The Auto mode (§5.2.5) keeps the layer in Mode C (always-flatten) most of the time.
- The forward output remains correct; only the per-head splines are operating at capacity.

This regime is the right behaviour: it tells the operator "this layer is under-provisioned for the data at the architecture's resolution." The diagnostic value of `ActiveHeads = H_max ∧ clip_rate ≫ τ_squaring` is high; it is a clear, observable signal that the model architecture itself is the bottleneck.

#### 5.4.9 Diagnostic Value of Final `ActiveHeads`

After a deployment has settled, the per-layer `ActiveHeads` distribution is a free readout of the layer's informational complexity on the observed data:

- Layers stuck at 2: simple, uniform attention patterns. Probably could have been smaller in the original architecture.
- Layers at 4–16: moderate complexity, multi-modal attention where it matters.
- Layers at 64+ or hitting `H_max`: doing genuinely complex discrimination work.

This is information not available from any standard transformer at any inspection level. It is a side-effect of the spec, not a designed feature, but it is genuinely useful.

#### 5.4.10 Invariants Maintained

After every per-inference pipeline step (§7), per layer:

- **(HS-1)** `ActiveHeads ∈ ℕ`, `2 ≤ ActiveHeads ≤ H_max`, monotonic non-decreasing across the run.
- **(HS-2)** `ActiveHeads` is a power of 2 (consequence of repeated squaring from the initial value 2). After clamping at `H_max`, this may not hold if `H_max` is not a power of 2; in that case `ActiveHeads ∈ {2, 4, 16, 256, …, H_max}`.
- **(HS-3)** Inactive heads (index `≥ ActiveHeads`) contribute zero to the layer's forward output.
- **(HS-4)** Squaring transitions are deterministic given the per-layer RNG state.

### 5.5 Autonomous Local Learning

#### 5.5.1 Premise — No Autograd at Inference

Inference must not require backpropagation, gradient tapes, or any optimiser machinery. The KAN evolves entirely through **autonomous local rules**: each spline coefficient updates from quantities locally observable at the layer (the score being processed, the basis activations, and the local residual). No external loss function is constructed; no gradient flows from anywhere outside the layer.

This constraint is satisfied by exploiting a structural property of B-splines: **`φ` is linear in its coefficients**. For linear-in-parameter models the locally optimal least-squares fit step has a closed form per observation (Normalised LMS / Widrow-Hoff), which can be expressed as a per-coefficient additive update with no gradient machinery.

Mathematically, this rule *is* equivalent to SGD-on-MSE for the linear regression problem; structurally, it is a **cellular update rule each coefficient applies to itself**, with no tape, no graph, and no external orchestrator. That structural distinction is what makes it "autonomous" rather than "SGD in disguise" — it slots into a pure-forward inference pass with no machinery extension.

Two rules exist, one per phase. The *form* is identical (NLMS); only the **target** differs.

#### 5.5.2 Phase M Rule — NLMS Toward `exp(s)`

For each attention score `s` evaluated in a forward pass, while the head's `status = SoftmaxActive`:

```
target := exp(s)                                          -- pre-normalisation softmax kernel
y_hat  := φ(s) = Σ_j c_j · B_j(s)
err    := target − y_hat

denom  := Σ_j B_j(s)² + δ                                  -- δ for stability, e.g. 1e-6
for each j with B_j(s) ≠ 0:                                -- only k+1 coefficients in support
    c_j ← c_j + η · B_j(s) · err / denom
```

**Why fit `exp(s)` rather than `softmax(s)`:** `softmax` involves the row-wise normalisation `/ Σ`, which is non-linear in `φ`'s coefficients (because the denominator depends on all coefficients). Fitting `exp(s)` decouples the local update from the row context — each `(s, exp(s))` pair is independently informative. After takeover, `φ(s) / Σφ(s)` then automatically matches softmax, since `exp(s) / Σexp(s) ≡ softmax(s)`.

**Why NLMS rather than plain LMS:** the denominator `Σ B_j(s)² + δ` makes the update **scale-invariant in the basis activations** at the observed score. This auto-tunes the per-step magnitude — a region of the spline with a high-magnitude basis hit gets the same effective step size as a low-magnitude one. Eliminates the need for a per-region learning-rate schedule.

**Locality of compute:** B-spline locality means only `k+1 = 4` coefficients have non-zero basis activation at any single score `s`. Each NLMS update touches only those `k+1` coefficients. Cost per score: O(`k+1`).

#### 5.5.3 Phase D Rule — Self-Distillation Sharpening

For each attention score `s` evaluated in a forward pass, while the head's `status = KANActive`:

```
w        := φ(s_i_·) / Σ_j φ(s_i_j)        -- current attention weights for the row containing s
w_sharp  := w^α / Σ(w^α)                    -- power-α renormalisation, α slightly > 1
target_s := w_sharp[position(s)] · Σ φ(s_i_·)    -- back into pre-normalisation space
err      := target_s − φ(s)

denom    := Σ_j B_j(s)² + δ
for each j with B_j(s) ≠ 0:
    c_j ← c_j + η · B_j(s) · err / denom
```

Same NLMS form as Phase M; only the target changes. The target is now derived from the head's **own current attention distribution**, sharpened by raising to a power `α` slightly greater than 1 and re-normalising.

**Default `α = 1.1`.** Each step is a gentle multiplicative emphasis of larger weights at the expense of smaller ones. Mechanism #1 + GM=1 enforce the budget; only the *redistribution* of mass is free.

**Why power-α rather than temperature-scaled softmax:** the temperature variant `softmax(log(w) / τ)` blows up numerically when any `w_i ≈ 0` (because `log w → −∞`). Power-α (`w^α / Σ w^α`) is well-defined for any non-negative `w`, including zeros. Since the spline output enforces `φ ≥ 0` (§5.1), `w` may legitimately be very small but is never undefined.

#### 5.5.4 Why Phase D Surfaces Information Rather Than Collapsing

The Phase D rule, run on its own without any constraint, would push the spline toward a one-hot distribution at the highest-scoring position — losing all attention information. **It does not, because of mechanism #1's product preservation and the GM=1 gauge.**

The dynamic is:

1. The Phase D rule wants to grow peaks (sharpen).
2. Mechanism #1 preserves per-window `Π |c_j|` (M1-1).
3. The GM=1 gauge fixes the overall product per head (GM-1).
4. Therefore peaks **cannot grow** in absolute coefficient magnitude — they can only grow in *relative* magnitude, by drawing mass away from other regions.

This forces a strict mass budget. The KAN cannot simply amplify what softmax already does. To sharpen anywhere, it must flatten elsewhere. The flattening is selective — the Phase D rule pulls mass from regions where the score signal is **inconsistent** (because the additive updates from inconsistent observations cancel) and concentrates it at regions where the score signal is **consistent** (because the updates reinforce).

**Net effect:** the spline learns a shape with multi-modal peaks at score values that consistently carry signal in the actual inference data distribution. Score regions that softmax would have crushed (low-score tokens carrying real information) get carved out as new peaks, drawing mass from regions of the spline that were previously redundant.

**This is the design's answer to "surface information that softmax would lose in a small model."** The combination of (a) shape freedom from B-spline parametrisation, (b) bounded budget from product preservation + gauge, (c) sharpening pressure from self-distillation, and (d) gradient-free local update produces a system that cannot grow without bound but will preferentially concentrate its mass on whatever consistently informative structure the data exhibits.

#### 5.5.5 Cost Analysis

Per forward pass with sequence length `L`, head count `H`, B-spline order `k`:

- **Spline evaluations:** `L²` per head (one per attention score), each O(`k+1`). Total `H · L² · (k+1)` multiply-adds.
- **NLMS updates:** `L²` per head, each O(`k+1`). Total `H · L² · (k+1)` multiply-adds. Same asymptotic cost as evaluation.
- **Mechanism #1 sweep:** `H · N · cascade_iters · (2k+1)` multiply-adds. Typically `cascade_iters ≤ 3`. Independent of `L`.
- **GM=1 gauge:** `H · N` log/exp operations. Independent of `L`.

For typical sequence lengths (`L ≫ N`), the per-pass cost is **dominated by the spline evaluations and updates, both linear in the existing attention's score-matrix cost**. The non-evaluation overhead (mechanism #1 + gauge) is O(`H · N`) per pass, asymptotically free relative to the attention's O(`L² · d`).

#### 5.5.6 Invariants Maintained

- **(LL-1)** Each NLMS update touches at most `k+1` coefficients per score — the ones with non-zero basis activation at `s`.
- **(LL-2)** The update is deterministic given `(s, target, denom, η, current coefficients)`.
- **(LL-3)** The update is sign-preserving in expectation (single-step updates can flip sign in pathological cases; mechanism #1 immediately corrects via low-lift).
- **(LL-4)** No gradient state, no optimiser state, no autograd graph is constructed at any point.








