# Delphi examples — KAN attention variants

Mirror tree of [`delphi/examples/`](../examples) holding examples that exercise
the **KAN attention** path in the Delphi library port. Each example is the same
program as its sibling under `delphi/examples/`, with the network class
swapped from `TNNet` / `THistoricalNets` to `TKANNet` and the transformer's
per-head softmax replaced by `TNNetKANNormaliser` via
`TKANNet.AddKANSelfAttention`.

## Status

The KAN library code under [`delphi/neural/`](../neural) is a work in progress.
Several `TNNetKANNormaliser` pipeline methods and the `AddKANSelfAttention`
builder body are still stubs that raise `EKANBadState` at runtime. **These
examples are pre-wired so that the moment the remaining stubs are filled in,
they become immediately runnable** — no further changes to the example
sources are required.

What's wired up already:
- `TKANNet` used as the network class (in place of `THistoricalNets` / `TNNet`).
- KAN units listed in the `uses` clause with explicit relative paths.
- Transformer blocks built by hand to mirror `TNNet.AddTransformerBlockCAI`,
  but with `AddKANSelfAttention` substituted in for the attention sub-chain.
- `LockToInference` called immediately after training so the final
  post-training generation engages the KAN attention path; in-training
  generation continues to use the bit-identical softmax fallback.

## Contents

| Path | Source | Notes |
|---|---|---|
| `SimpleNLP/SimpleTransformer.dpr` | mirrors `examples/SimpleNLP/SimpleTransformer.dpr` | TinyStories character-level transformer; the only example in the tree that has attention layers to KAN-ify. |

Other examples (image classifiers, MLP toys, classifier-only convolutional
nets) have no self-attention path and so don't have a KAN variant. If/when a
future example uses self-attention it gets the same treatment here.

## How to build

Open the `.dpr` directly in Delphi. The `uses` clause carries explicit
relative paths (`..\..\neural\...` for library units, `..\..\examples\CustApp.pas`
for the `TCustomApplication` shim), so no library-path setup is needed.
Datasets and model files are loaded from the working directory, exactly as in
the parent `delphi/examples/` tree.
