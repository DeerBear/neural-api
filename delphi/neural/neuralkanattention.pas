(*
neuralkanattention
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

unit neuralkanattention;
// =====================================================================
// STANDALONE DELPHI PORT (modern Delphi 10.x - 12 Athens).
// Hand-converted from ../../neural/neuralkanattention.pas via ../_port_tool.py, which
// is Pascal-lexer-aware (directives in comments/strings untouched). FPC
// and the AVX defines are resolved undefined; FPC-only branches give
// way to the upstream Delphi branch. Canonical float/pointer types come
// from neuralsimd, re-exported through neuralvolume so call sites are
// unchanged. Review-verified only -- no Delphi toolchain was available.
// =====================================================================


(*
KAN attention network class. Owns the per-attention-layer metadata
(basis, RNG, normaliser registry, squaring trigger) and the network-wide
mode lock. Provides AddKANSelfAttention as the construction entry point.

Mechanism specification:    docs/kan_attention_spec.md §3, §11, §14
Implementation decisions:   docs/kan_implementation_pascal.md §4, §6

v1 status: SKELETON. The AddKANSelfAttention builder, squaring trigger,
and telemetry dump are stubs raising EKANBadState; mode-safety guards
and bulk enable/disable are real.
*)


interface

uses
  Classes, SysUtils, Math,
  neuralvolume, neuralnetwork, neuralfit,
  neuralkantypes, neuralkanbasis, neuralkannormaliser;

type
  // ===================================================================
  //  PER-ATTENTION-LAYER METADATA
  // ===================================================================

  /// Tracks the H_max normalisers and shared resources for one
  /// attention layer in a TKANNet. Owned by the network.
  ///
  /// Per-attention-layer state covered:
  ///   - the shared B-spline basis and seeded RNG (one of each per layer)
  ///   - the registry of H_max TNNetKANNormaliser instances (one per head sub-path)
  ///   - the squaring trigger machinery: clip-rate EMA and the
  ///     consecutive-high-sweeps counter (spec §5.4.3)
  TKANAttentionLayerInfo = class
  private
    FAttentionLayerId: integer;
    FBasis: TKANBasis;
    FRNG: TKANSeededRNG;

    // --- Pre-allocation and squaring (spec §5.4) ---
    FHMax: integer;
    FActiveHeads: integer;

    // --- Squaring trigger machinery (spec §5.4.3) ---
    FClipRateEMA: TNeuralFloat;            // EMA of per-sweep clip-or-lift fraction
    FClipRateEmaLambda: TNeuralFloat;      // EMA decay; default 0.01 (spec §6.2 horizon)
    FTauSquaring: TNeuralFloat;            // clip-rate threshold; default 0.10
    FKSquaring: integer;                   // sustained sweeps required; default 64
    FConsecutiveHighSweeps: integer;       // current run of sweeps above τ_squaring

    FNormalisers: TList;               // of TNNetKANNormaliser; length = H_max

    // Q/K/V projection layer references. Captured at AddKANSelfAttention time
    // so the session-level weight clipper can apply a looser threshold to
    // attention projections than to feedforward layers. Without this the
    // global clip ceiling (e.g. ±0.20) caps Q/K weights at a level where
    // dot products stay near zero, yielding degenerate uniform attention.
    FQProjection, FKProjection, FVProjection: TNNetLayer;
  public
    constructor Create(const AttentionLayerId: integer;
                       const Spec: TKANGridSpec;
                       const HMax: integer;
                       const Seed: UInt64;
                       const TauSquaring: TNeuralFloat = 0.10;
                       const KSquaring: integer = 64;
                       const ClipRateEmaLambda: TNeuralFloat = 0.01;
                       const InitialActiveHeads: integer = 2); reintroduce;
    destructor Destroy; override;

    procedure RegisterNormaliser(const N: TNNetKANNormaliser);

    /// Register the Q/K/V projection layer references created in
    /// AddKANSelfAttention. Used by the session-level clipper to apply a
    /// looser threshold to these specific layers.
    procedure RegisterQKVProjections(const Q, K, V: TNNetLayer);

    /// Update the per-layer FClipRateEMA from the fraction of coefficients
    /// that fired clip-or-lift in the most recent mechanism-#1 sweep
    /// (across all *active* heads in this attention layer). Also updates
    /// FConsecutiveHighSweeps according to whether the new EMA is above
    /// FTauSquaring. Call once per Compute pass per attention layer
    /// (spec §5.2.7, §5.4.3).
    procedure RecordSweepClipRate(const SweepClipRate: TNeuralFloat);

    /// Returns true if the squaring criterion is currently met:
    ///   FActiveHeads < FHMax  AND  FConsecutiveHighSweeps >= FKSquaring.
    /// The caller (typically TKANNet at §7 step 11) is responsible for
    /// having previously called RecordSweepClipRate to update the running
    /// counters. Returns false once FActiveHeads = FHMax (no more room)
    /// or while pressure has not been sustained for FKSquaring sweeps.
    function ShouldFireSquaring: boolean;

    /// Per-attention-layer post-batch hook: checks the squaring criterion
    /// and fires squaring if met. To be called by TKANNet at §7 step 11.
    /// (The actual squaring operation — log-normal perturbation of newly-
    /// active heads — is still a stub.)
    procedure CheckSquaring;

    property AttentionLayerId: integer read FAttentionLayerId;
    property Basis: TKANBasis read FBasis;
    property HMax: integer read FHMax;
    property ActiveHeads: integer read FActiveHeads;
    property ClipRateEMA: TNeuralFloat read FClipRateEMA;
    property ConsecutiveHighSweeps: integer read FConsecutiveHighSweeps;
    property TauSquaring: TNeuralFloat read FTauSquaring;
    property KSquaring: integer read FKSquaring;
    property Normalisers: TList read FNormalisers;
    property QProjection: TNNetLayer read FQProjection;
    property KProjection: TNNetLayer read FKProjection;
    property VProjection: TNNetLayer read FVProjection;
  end;

  // ===================================================================
  //  NETWORK CLASS  (spec §3, §11, §14; impl doc §4, §6)
  // ===================================================================

  /// Network subclass that owns the KAN attention infrastructure:
  /// registry of KAN attention layers, mode lock, bulk operations, and
  /// the AddKANSelfAttention builder.
  TKANNet = class(TNNet)
  private
    FInferenceLocked: boolean;
    FAttentionLayers: TList;           // of TKANAttentionLayerInfo
    FNextAttentionLayerId: integer;

    procedure AssertNotLocked(const OpName: string);
    procedure AssertLocked(const OpName: string);
  public
    constructor Create; override;
    destructor Destroy; override;

    /// Builds an attention chain with TNNetKANNormaliser substituted for
    /// TNNetPointwiseSoftMax. Pre-allocates HMax head sub-paths; only
    /// InitialHeads are active at start (others masked to zero per HS-3).
    function AddKANSelfAttention(
      const InitialHeads: integer = 2;
      const HeadCeiling: integer = 0;          // 0 -> auto: largest pow2 <= d/2
      const GridLow: TNeuralFloat = -8.0;
      const GridHigh: TNeuralFloat = 8.0;
      const GridKnots: integer = 64;
      const BasisOrder: integer = 3;
      const SharpenAlpha: TNeuralFloat = 1.1;
      const KLThreshold: TNeuralFloat = 0.01;
      const KLConfirmPasses: integer = 100;
      const SquaringClipRate: TNeuralFloat = 0.10;
      const SquaringSweeps: integer = 64
    ): TNNetLayer;

    /// One-way commit to inference. Sets FInferenceLocked := true and
    /// propagates InferenceMode to every registered KAN normaliser.
    /// Idempotent.
    procedure LockToInference;

    /// Inference-only forward path. Asserts FInferenceLocked.
    procedure InferenceForward(pInput: TNNetVolume);

    // --- Bulk operations on KAN layers ---
    procedure DisableKANForLayer(const Idx: integer);
    procedure EnableKANForLayer(const Idx: integer);
    procedure DisableAllKAN;
    procedure EnableAllKAN;
    function  KANEnabledMask: TKANEnabledMask;

    // --- Telemetry ---
    function  KANTelemetry: string;

    // --- Coefficient snapshot/restore (non-destructive evaluation) ---
    /// Snapshot the FHead.Coeffs of every TNNetKANNormaliser across all
    /// attention layers (depth-first: layer 0 head 0 .. layer 0 head H-1,
    /// layer 1 head 0, ...). Pair with RestoreAllCoeffs to undo any
    /// coefficient drift caused by inference-mode forward passes.
    function  SnapshotAllCoeffs: TKANCoeffSnapshot;
    /// Restore from a snapshot produced by SnapshotAllCoeffs. Length must
    /// match the current normaliser count; raises EKANBadState otherwise.
    procedure RestoreAllCoeffs(const Snapshot: TKANCoeffSnapshot);

    // --- Post-lock alpha auto-tuner (spec extension; not in original spec) ---
    /// Set SharpenAlpha on every TNNetKANNormaliser across all attention layers.
    /// Bulk operation used by CalibrateAlpha and available for manual override.
    procedure SetAllAlpha(const Value: TNeuralFloat);

    /// Read the current SharpenAlpha. Returns the value from the first
    /// normaliser in the first attention layer (all are kept in sync by
    /// SetAllAlpha; if the caller has set them per-head manually this is
    /// the wrong accessor).
    function  GetCurrentAlpha: TNeuralFloat;

    /// Continuous EMA-driven SharpenAlpha calibrator. Runs a finite-difference
    /// gradient descent on validation cross-entropy loss and writes the
    /// selected alpha to every normaliser. Final alpha selection: by default
    /// the alpha at the iteration with the lowest observed mean loss
    /// (UseBestAlpha=true). Set UseBestAlpha=false to use the EMA-smoothed
    /// trajectory endpoint, the previous behaviour. The EMA picker was prone
    /// to landing on a value that none of the iterations actually scored
    /// well at, when the noisy gradient pushed the trajectory away from the
    /// observed minimum (the original symptom that motivated this rewrite).
    ///
    /// Must be called post-LockToInference. Every forward pass through a
    /// TNNetKANNormaliser in inference mode runs a NLMS Phase M/D update
    /// that mutates FHead.Coeffs (see TNNetKANNormaliser.Compute step 7);
    /// across the 3*SamplesPerIter*Iterations forward passes the calibrator
    /// performs, this drift is large enough to wash out the original
    /// coefficients. PreserveCoeffs (default true) snapshots FHead.Coeffs
    /// before the sweep and restores it after, so the only persistent
    /// effect is the SharpenAlpha value. Set PreserveCoeffs=false to let
    /// the calibration act as adaptive fine-tuning on validation data.
    ///
    /// Each iteration:
    ///   1. Sample SamplesPerIter validation examples.
    ///   2. For each: forward 3x at alpha-AlphaStep, alpha, alpha+AlphaStep.
    ///   3. Estimate dLoss/dAlpha via central finite difference.
    ///   4. SGD step on alpha (clipped to [AlphaMin, AlphaMax]).
    ///   5. EMA-smooth: alpha_ema := (1-EMABeta)*alpha_ema + EMABeta*alpha.
    ///   6. If MeanLoss < best so far, record (alpha, loss).
    /// Final write uses the alpha selected per UseBestAlpha.
    ///
    /// Cost: 3 * SamplesPerIter * Iterations forward passes. With defaults
    /// (Iterations=20, SamplesPerIter=16) that's 960 forwards -- roughly
    /// 1/5 of a full validation pass.
    procedure CalibrateAlpha(
      GetValidationPair: TNNetGet2VolumesProc;
      const ValidationCount: integer;
      const Iterations: integer = 20;
      const SamplesPerIter: integer = 16;
      const AlphaMin: TNeuralFloat = 0.5;
      const AlphaMax: TNeuralFloat = 3.0;
      const AlphaStep: TNeuralFloat = 0.05;
      const LearningRate: TNeuralFloat = 0.2;
      const EMABeta: TNeuralFloat = 0.3;
      const PreserveCoeffs: boolean = true;
      const UseBestAlpha: boolean = true
    );

    // --- Training-method gate on lock state ---
    // Parent TNNet.Backpropagate(pOutput: TNNetVolume) is overload but not
    // virtual, so we can't override it. reintroduce; overload; shadows it
    // for calls through a TKANNet reference; calls through a TNNet
    // reference bypass our assertion (defence-in-depth only -- the per-
    // layer normaliser has its own lock check in Backpropagate).
    procedure Backpropagate(pInput: TNNetVolume); reintroduce; overload;

    property InferenceLocked: boolean read FInferenceLocked;
    property AttentionLayers: TList read FAttentionLayers;
  end;

implementation

// =====================================================================
//  TKANAttentionLayerInfo
// =====================================================================

constructor TKANAttentionLayerInfo.Create(const AttentionLayerId: integer;
                                           const Spec: TKANGridSpec;
                                           const HMax: integer;
                                           const Seed: UInt64;
                                           const TauSquaring: TNeuralFloat;
                                           const KSquaring: integer;
                                           const ClipRateEmaLambda: TNeuralFloat;
                                           const InitialActiveHeads: integer);
begin
  inherited Create;

  if HMax < 2 then
    raise EKANBadState.CreateFmt(
      'TKANAttentionLayerInfo.Create: HMax must be >= 2 (got %d)', [HMax]);
  if (TauSquaring <= 0) or (TauSquaring >= 1) then
    raise EKANBadState.CreateFmt(
      'TKANAttentionLayerInfo.Create: TauSquaring must be in (0,1), got %g', [TauSquaring]);
  if KSquaring < 1 then
    raise EKANBadState.CreateFmt(
      'TKANAttentionLayerInfo.Create: KSquaring must be >= 1 (got %d)', [KSquaring]);
  if (ClipRateEmaLambda <= 0) or (ClipRateEmaLambda > 1) then
    raise EKANBadState.CreateFmt(
      'TKANAttentionLayerInfo.Create: ClipRateEmaLambda must be in (0,1], got %g',
      [ClipRateEmaLambda]);
  if (InitialActiveHeads < 2) or (InitialActiveHeads > HMax) then
    raise EKANBadState.CreateFmt(
      'TKANAttentionLayerInfo.Create: InitialActiveHeads %d must be in [2, HMax=%d]',
      [InitialActiveHeads, HMax]);

  FAttentionLayerId := AttentionLayerId;
  FBasis := TKANBasis.Create(Spec);
  FRNG.Seed(Seed);

  FHMax := HMax;
  FActiveHeads := InitialActiveHeads;   // spec §5.4.2 default is 2; growth via squaring.

  FClipRateEMA := 0;
  FClipRateEmaLambda := ClipRateEmaLambda;
  FTauSquaring := TauSquaring;
  FKSquaring := KSquaring;
  FConsecutiveHighSweeps := 0;

  FNormalisers := TList.Create;
end;

destructor TKANAttentionLayerInfo.Destroy;
begin
  FreeAndNil(FBasis);
  FreeAndNil(FNormalisers);   // does not free the normalisers themselves
  inherited Destroy;
end;

procedure TKANAttentionLayerInfo.RegisterNormaliser(const N: TNNetKANNormaliser);
begin
  FNormalisers.Add(N);
end;

procedure TKANAttentionLayerInfo.RegisterQKVProjections(const Q, K, V: TNNetLayer);
begin
  FQProjection := Q;
  FKProjection := K;
  FVProjection := V;
end;

procedure TKANAttentionLayerInfo.RecordSweepClipRate(
  const SweepClipRate: TNeuralFloat);
begin
  // EMA update: FClipRateEMA <- (1 - λ) · FClipRateEMA + λ · SweepClipRate.
  FClipRateEMA := (1 - FClipRateEmaLambda) * FClipRateEMA
                  + FClipRateEmaLambda * SweepClipRate;

  // Sustained-pressure counter: increment if the EMA is above the threshold,
  // otherwise reset to zero. The reset on a single low sweep is intentional —
  // K_squaring (default 64) is what gives the sustained-pressure semantics.
  if FClipRateEMA > FTauSquaring then
    Inc(FConsecutiveHighSweeps)
  else
    FConsecutiveHighSweeps := 0;
end;

function TKANAttentionLayerInfo.ShouldFireSquaring: boolean;
begin
  Result := (FActiveHeads < FHMax)
            and (FConsecutiveHighSweeps >= FKSquaring);
end;

procedure TKANAttentionLayerInfo.CheckSquaring;
begin
  if ShouldFireSquaring then
  begin
    // TODO: actual squaring operation per spec §5.4.4 — log-normal
    // perturbation of newly-active heads using FRNG. For now this is
    // a stub; the trigger detection itself is real and tested.
    raise EKANBadState.Create(
      'TKANAttentionLayerInfo.CheckSquaring: trigger fired but DoSquaring not implemented');
    // FConsecutiveHighSweeps := 0;   // would reset after firing
  end;
end;

// =====================================================================
//  TKANNet
// =====================================================================

constructor TKANNet.Create;
begin
  inherited Create;
  FInferenceLocked := false;
  FAttentionLayers := TList.Create;
  FNextAttentionLayerId := 0;
end;

destructor TKANNet.Destroy;
var
  i: integer;
begin
  if Assigned(FAttentionLayers) then
  begin
    for i := 0 to FAttentionLayers.Count - 1 do
      TKANAttentionLayerInfo(FAttentionLayers[i]).Free;
    FreeAndNil(FAttentionLayers);
  end;
  inherited Destroy;
end;

procedure TKANNet.AssertNotLocked(const OpName: string);
begin
  if FInferenceLocked then
    raise EKANInInference.CreateFmt(
      '%s is not permitted: network is locked to inference. ' +
      'Construct a new TKANNet from a checkpoint to resume training.',
      [OpName]);
end;

procedure TKANNet.AssertLocked(const OpName: string);
begin
  if not FInferenceLocked then
    raise EKANNotLocked.CreateFmt(
      '%s requires LockToInference to have been called first.',
      [OpName]);
end;

function TKANNet.AddKANSelfAttention(
  const InitialHeads: integer;
  const HeadCeiling: integer;
  const GridLow: TNeuralFloat;
  const GridHigh: TNeuralFloat;
  const GridKnots: integer;
  const BasisOrder: integer;
  const SharpenAlpha: TNeuralFloat;
  const KLThreshold: TNeuralFloat;
  const KLConfirmPasses: integer;
  const SquaringClipRate: TNeuralFloat;
  const SquaringSweeps: integer
): TNNetLayer;
var
  PreviousLayer: TNNetLayer;
  PreviousDepth, PreviousSizeX, PreviousSizeY, InputChannelsPerGroup: integer;
  HMax, HeadCnt, AttentionLayerId: integer;
  Spec: TKANGridSpec;
  Info: TKANAttentionLayerInfo;
  Normaliser: TNNetKANNormaliser;
  Seed: UInt64;
  QueryGroup, KeyGroup, ValueGroup: TNNetLayer;
  LocalQueryGroup, LocalKeyGroup, LocalValueGroup, LocalValueTGroup: TNNetLayer;
  NormaliserLayer: TNNetLayer;
  EachGroupOutput: array of TNNetLayer;

  function LargestPow2AtMost(const n: integer): integer;
  begin
    Result := 1;
    while (Result * 2) <= n do Result := Result * 2;
  end;

begin
  AssertNotLocked('AddKANSelfAttention');

  PreviousLayer := GetLastLayer();
  PreviousDepth := PreviousLayer.Output.Depth;
  PreviousSizeX := PreviousLayer.Output.SizeX;
  PreviousSizeY := PreviousLayer.Output.SizeY;
  if PreviousSizeY > 1 then
    PreviousLayer := AddLayer(TNNetReshape.Create(
      PreviousSizeX * PreviousSizeY, 1, PreviousDepth));

  // (1) Resolve H_max. Spec §5.4.2: default is the largest power of two
  // not exceeding floor(d/2). Caller can override with an explicit ceiling.
  if HeadCeiling = 0 then
    HMax := LargestPow2AtMost(PreviousDepth div 2)
  else
    HMax := HeadCeiling;
  if HMax < 2 then HMax := 2;
  if (PreviousDepth mod HMax) <> 0 then
    raise EKANBadState.CreateFmt(
      'AddKANSelfAttention: PreviousDepth %d not divisible by HMax %d',
      [PreviousDepth, HMax]);
  if (InitialHeads < 2) or (InitialHeads > HMax) then
    raise EKANBadState.CreateFmt(
      'AddKANSelfAttention: InitialHeads %d must be in [2, HMax=%d]',
      [InitialHeads, HMax]);
  if SharpenAlpha <= 1.0 then
    raise EKANBadState.CreateFmt(
      'AddKANSelfAttention: SharpenAlpha %g must be > 1 (spec §5.5.3)',
      [SharpenAlpha]);
  if KLThreshold <= 0 then
    raise EKANBadState.CreateFmt(
      'AddKANSelfAttention: KLThreshold %g must be > 0 (nats, spec §6.3)',
      [KLThreshold]);
  if KLConfirmPasses < 1 then
    raise EKANBadState.CreateFmt(
      'AddKANSelfAttention: KLConfirmPasses %d must be >= 1 (spec §6.3)',
      [KLConfirmPasses]);

  // (2) Build grid spec and per-layer info. Seed combines a per-network
  // counter with the grid hash so two layers with the same spec still get
  // distinct deterministic streams (spec §9).
  Spec.GridLow := GridLow;
  Spec.GridHigh := GridHigh;
  Spec.KnotCount := GridKnots;
  Spec.BasisOrder := BasisOrder;
  AttentionLayerId := FNextAttentionLayerId;
  Inc(FNextAttentionLayerId);
  // Unsigned wraparound is intentional; disable Delphi's default
  // {$Q+}/{$R+} just for this expression.
  {$Q-}{$R-}
  Seed := (UInt64(AttentionLayerId + 1) * UInt64($9E3779B97F4A7C15)) xor Spec.Hash;
  {$Q+}{$R+}
  if Seed = 0 then Seed := 1;

  Info := TKANAttentionLayerInfo.Create(AttentionLayerId, Spec, HMax, Seed,
    SquaringClipRate, SquaringSweeps, 0.01, InitialHeads);
  FAttentionLayers.Add(Info);

  // (3) Q/K/V projections — mirror AddSelfAttention's no-HasNorm branch.
  // Capture the projection layer references in Info before reassigning the
  // *Group variables to the downstream SqrtRoot layers; the session-level
  // clipper queries Info.QProjection/KProjection/VProjection to apply a
  // looser weight clip to the attention projections.
  QueryGroup := AddLayerAfter([TNNetPointwiseConvLinear.Create(PreviousDepth, 1)], PreviousLayer);
  KeyGroup   := AddLayerAfter([TNNetPointwiseConvLinear.Create(PreviousDepth, 1)], PreviousLayer);
  ValueGroup := AddLayerAfter([TNNetPointwiseConvLinear.Create(PreviousDepth, 1)], PreviousLayer);
  Info.RegisterQKVProjections(QueryGroup, KeyGroup, ValueGroup);
  QueryGroup := AddLayerAfter([TNNetSignedSquareRoot1.Create()], QueryGroup);
  KeyGroup   := AddLayerAfter([TNNetSignedSquareRoot1.Create()], KeyGroup);
  ValueGroup := AddLayerAfter([TNNetSignedSquareRoot1.Create()], ValueGroup);

  // (4) Per-head sub-path. The TNNetKANNormaliser sits immediately downstream
  // of the per-head TNNetPointwiseSoftMax as a chain-level decorator. While
  // the network is unlocked (or KAN is disabled), the decorator is a pure
  // identity passthrough, so the forward+backward path is bit-identical to
  // AddSelfAttention — this is what gives IT-Retrofit.
  SetLength(EachGroupOutput, HMax);
  InputChannelsPerGroup := PreviousDepth div HMax;
  for HeadCnt := 0 to HMax - 1 do
  begin
    LocalQueryGroup := AddLayerAfter([
      TNNetSplitChannels.Create(HeadCnt * InputChannelsPerGroup, InputChannelsPerGroup)],
      QueryGroup);
    LocalKeyGroup := AddLayerAfter([
      TNNetSplitChannels.Create(HeadCnt * InputChannelsPerGroup, InputChannelsPerGroup)],
      KeyGroup);
    LocalValueGroup := AddLayerAfter([
      TNNetSplitChannels.Create(HeadCnt * InputChannelsPerGroup, InputChannelsPerGroup)],
      ValueGroup);
    LocalValueTGroup := AddLayerAfter(TNNetTransposeXD.Create(), LocalValueGroup);

    AddLayer(TNNetDotProducts.Create(LocalQueryGroup, LocalKeyGroup, {NoForward=}false));
    AddLayer(TNNetMulByConstant.Create(1 / Sqrt(InputChannelsPerGroup)));
    AddLayer(TNNetReLUL.Create(-500, +500, 0));
    AddLayer(TNNetPointwiseSoftMax.Create(0, 0));

    Normaliser := TNNetKANNormaliser.Create(Spec, AttentionLayerId, HeadCnt,
                                            Info.Basis, @Info.FRNG);
    Normaliser.SharpenAlpha := SharpenAlpha;
    Normaliser.KLThreshold := KLThreshold;
    Normaliser.KLConfirmPasses := KLConfirmPasses;
    NormaliserLayer := AddLayer(Normaliser);
    Info.RegisterNormaliser(Normaliser);

    AddLayer(TNNetDotProducts.Create(LocalValueTGroup, NormaliserLayer));
    EachGroupOutput[HeadCnt] := GetLastLayer();
  end;

  // (5) Concatenate heads and apply the output projection W_O.
  AddLayer(TNNetDeepConcat.Create(EachGroupOutput));
  SetLength(EachGroupOutput, 0);
  Result := AddLayer([TNNetPointwiseConvLinear.Create(PreviousDepth, 1)]);
  Result := AddLayer([TNNetSignedSquareRoot1.Create()]);

  if PreviousSizeY > 1 then
    Result := AddLayer(TNNetReshape.Create(PreviousSizeX, PreviousSizeY, PreviousDepth));
end;

procedure TKANNet.LockToInference;
var
  i, j: integer;
  Info: TKANAttentionLayerInfo;
  Normaliser: TNNetKANNormaliser;
begin
  if FInferenceLocked then exit;     // idempotent

  for i := 0 to FAttentionLayers.Count - 1 do
  begin
    Info := TKANAttentionLayerInfo(FAttentionLayers[i]);
    for j := 0 to Info.Normalisers.Count - 1 do
    begin
      Normaliser := TNNetKANNormaliser(Info.Normalisers[j]);
      Normaliser.EnterInferenceMode;
    end;
  end;

  FInferenceLocked := true;
end;

procedure TKANNet.InferenceForward(pInput: TNNetVolume);
begin
  AssertLocked('InferenceForward');
  inherited Compute(pInput);
end;

procedure TKANNet.DisableKANForLayer(const Idx: integer);
var
  Info: TKANAttentionLayerInfo;
  j: integer;
begin
  if (Idx < 0) or (Idx >= FAttentionLayers.Count) then
    raise EKANBadState.CreateFmt('DisableKANForLayer: index %d out of range', [Idx]);
  Info := TKANAttentionLayerInfo(FAttentionLayers[Idx]);
  for j := 0 to Info.Normalisers.Count - 1 do
    TNNetKANNormaliser(Info.Normalisers[j]).KANEnabled := false;
end;

procedure TKANNet.EnableKANForLayer(const Idx: integer);
var
  Info: TKANAttentionLayerInfo;
  j: integer;
begin
  if (Idx < 0) or (Idx >= FAttentionLayers.Count) then
    raise EKANBadState.CreateFmt('EnableKANForLayer: index %d out of range', [Idx]);
  Info := TKANAttentionLayerInfo(FAttentionLayers[Idx]);
  for j := 0 to Info.Normalisers.Count - 1 do
    TNNetKANNormaliser(Info.Normalisers[j]).KANEnabled := true;
end;

procedure TKANNet.DisableAllKAN;
var
  i: integer;
begin
  for i := 0 to FAttentionLayers.Count - 1 do DisableKANForLayer(i);
end;

procedure TKANNet.EnableAllKAN;
var
  i: integer;
begin
  for i := 0 to FAttentionLayers.Count - 1 do EnableKANForLayer(i);
end;

function TKANNet.KANEnabledMask: TKANEnabledMask;
var
  i: integer;
  Info: TKANAttentionLayerInfo;
begin
  SetLength(Result, FAttentionLayers.Count);
  for i := 0 to FAttentionLayers.Count - 1 do
  begin
    Info := TKANAttentionLayerInfo(FAttentionLayers[i]);
    // True if any normaliser in the layer is enabled. (All H_max within
    // a layer should be in lockstep since the bulk ops set them together.)
    Result[i] := (Info.Normalisers.Count > 0)
                 and TNNetKANNormaliser(Info.Normalisers[0]).KANEnabled;
  end;
end;

function TKANNet.KANTelemetry: string;
// Per-attention-layer diagnostic dump (spec §14.5). Aggregates per-head
// state into one line per attention layer, suitable for logging at
// arbitrary cadence post-LockToInference. Format example:
//
//   KAN telemetry (2 attention layer(s)):
//     Layer 0: heads=16/16 status=14SM/2KAN KLema_avg=0.0083 clipRate=0.0021 capHits=0
//     Layer 1: heads=16/16 status=16SM/0KAN KLema_avg=inf    clipRate=0.0000 capHits=0
//
// KLema_avg is the mean across heads whose KLEMA has been initialised
// past the Infinity cold-start sentinel; if every head is still at the
// sentinel, "inf" is reported instead. CascadeCapHits is summed across
// all heads in the layer (rather than averaged) so a single misbehaving
// head is visible at the layer level. Pre-LockToInference everything
// reads as the cold-start defaults; the interesting numbers appear
// after some inference passes have run.
var
  i, j: integer;
  Info: TKANAttentionLayerInfo;
  N: TNNetKANNormaliser;
  SMCount, KANCount, FiniteKLCount, TotalCapHits: integer;
  KLSum: TNeuralFloat;
  KLAvgStr: string;
begin
  Result := Format('KAN telemetry (%d attention layer(s)):' + sLineBreak,
                   [FAttentionLayers.Count]);
  for i := 0 to FAttentionLayers.Count - 1 do
  begin
    Info := TKANAttentionLayerInfo(FAttentionLayers[i]);
    SMCount := 0;
    KANCount := 0;
    FiniteKLCount := 0;
    TotalCapHits := 0;
    KLSum := 0;
    for j := 0 to Info.Normalisers.Count - 1 do
    begin
      N := TNNetKANNormaliser(Info.Normalisers[j]);
      if N.Status = ksSoftmaxActive then Inc(SMCount) else Inc(KANCount);
      if not IsInfinite(N.KLEMA) then
      begin
        KLSum := KLSum + N.KLEMA;
        Inc(FiniteKLCount);
      end;
      TotalCapHits := TotalCapHits + N.CascadeCapHits;
    end;
    if FiniteKLCount > 0 then
      KLAvgStr := Format('%.4f', [KLSum / FiniteKLCount])
    else
      KLAvgStr := 'inf';
    Result := Result + Format(
      '  Layer %d: heads=%d/%d status=%dSM/%dKAN KLema_avg=%s clipRate=%.4f capHits=%d' + sLineBreak,
      [Info.AttentionLayerId, Info.ActiveHeads, Info.HMax,
       SMCount, KANCount, KLAvgStr, Info.ClipRateEMA, TotalCapHits]);
  end;
end;

procedure TKANNet.Backpropagate(pInput: TNNetVolume);
begin
  AssertNotLocked('Backpropagate');
  inherited Backpropagate(pInput);
end;

// =====================================================================
//  Post-lock alpha auto-tuner
// =====================================================================

procedure TKANNet.SetAllAlpha(const Value: TNeuralFloat);
var
  i, j: integer;
  Info: TKANAttentionLayerInfo;
begin
  for i := 0 to FAttentionLayers.Count - 1 do
  begin
    Info := TKANAttentionLayerInfo(FAttentionLayers[i]);
    for j := 0 to Info.Normalisers.Count - 1 do
      TNNetKANNormaliser(Info.Normalisers[j]).SharpenAlpha := Value;
  end;
end;

function TKANNet.SnapshotAllCoeffs: TKANCoeffSnapshot;
var
  i, j, Idx, Total: integer;
  Info: TKANAttentionLayerInfo;
begin
  Total := 0;
  for i := 0 to FAttentionLayers.Count - 1 do
    Total := Total + TKANAttentionLayerInfo(FAttentionLayers[i]).Normalisers.Count;
  SetLength(Result, Total);
  Idx := 0;
  for i := 0 to FAttentionLayers.Count - 1 do
  begin
    Info := TKANAttentionLayerInfo(FAttentionLayers[i]);
    for j := 0 to Info.Normalisers.Count - 1 do
    begin
      Result[Idx] := TNNetKANNormaliser(Info.Normalisers[j]).SnapshotCoeffs;
      Inc(Idx);
    end;
  end;
end;

procedure TKANNet.RestoreAllCoeffs(const Snapshot: TKANCoeffSnapshot);
var
  i, j, Idx, Total: integer;
  Info: TKANAttentionLayerInfo;
begin
  Total := 0;
  for i := 0 to FAttentionLayers.Count - 1 do
    Total := Total + TKANAttentionLayerInfo(FAttentionLayers[i]).Normalisers.Count;
  if Length(Snapshot) <> Total then
    raise EKANBadState.CreateFmt(
      'TKANNet.RestoreAllCoeffs: normaliser count mismatch (snapshot=%d, current=%d)',
      [Length(Snapshot), Total]);
  Idx := 0;
  for i := 0 to FAttentionLayers.Count - 1 do
  begin
    Info := TKANAttentionLayerInfo(FAttentionLayers[i]);
    for j := 0 to Info.Normalisers.Count - 1 do
    begin
      TNNetKANNormaliser(Info.Normalisers[j]).RestoreCoeffs(Snapshot[Idx]);
      Inc(Idx);
    end;
  end;
end;

function TKANNet.GetCurrentAlpha: TNeuralFloat;
var
  Info: TKANAttentionLayerInfo;
begin
  if (FAttentionLayers.Count = 0) then
    raise EKANBadState.Create('GetCurrentAlpha: no attention layers');
  Info := TKANAttentionLayerInfo(FAttentionLayers[0]);
  if (Info.Normalisers.Count = 0) then
    raise EKANBadState.Create('GetCurrentAlpha: no normalisers in layer 0');
  Result := TNNetKANNormaliser(Info.Normalisers[0]).SharpenAlpha;
end;

procedure TKANNet.CalibrateAlpha(
  GetValidationPair: TNNetGet2VolumesProc;
  const ValidationCount: integer;
  const Iterations: integer;
  const SamplesPerIter: integer;
  const AlphaMin: TNeuralFloat;
  const AlphaMax: TNeuralFloat;
  const AlphaStep: TNeuralFloat;
  const LearningRate: TNeuralFloat;
  const EMABeta: TNeuralFloat;
  const PreserveCoeffs: boolean;
  const UseBestAlpha: boolean
);
// Continuous EMA-driven SharpenAlpha calibration. See interface docstring
// for algorithm. The implementation is single-threaded for simplicity; the
// total cost is ~3*SamplesPerIter*Iterations forward passes which is small
// compared to a full validation pass (1/5 with defaults).
//
// Loss function mirrors TNeuralDataLoadingFit.DefaultLossFn:
// classification cross-entropy = -ln(output[expected.Tag]).
var
  InputVol, ExpectedVol: TNNetVolume;
  Iter, S, Idx, ClassId: integer;
  AlphaInitial, AlphaCenter, AlphaEMA, AlphaUp, AlphaDown: TNeuralFloat;
  AlphaBest, FinalAlpha, MeanLoss, LossBest: TNeuralFloat;
  LossSumCenter, LossSumUp, LossSumDown: TNeuralFloat;
  Gradient, NewAlpha, OutputValue: TNeuralFloat;
  LastLayer: TNNetLayer;
  CoeffSnapshot: TKANCoeffSnapshot;

  procedure AccumulateLoss(const Alpha: TNeuralFloat; var LossSum: TNeuralFloat);
  begin
    SetAllAlpha(Alpha);
    Compute(InputVol);
    ClassId := ExpectedVol.Tag;
    OutputValue := LastLayer.Output.FData[ClassId];
    if OutputValue > 0 then
      LossSum := LossSum + (-Ln(OutputValue))
    else
      LossSum := LossSum + 100;
  end;

begin
  AssertLocked('CalibrateAlpha');
  if FAttentionLayers.Count = 0 then exit;
  if not Assigned(GetValidationPair) then
    raise EKANBadState.Create('CalibrateAlpha: GetValidationPair is nil');
  if ValidationCount <= 0 then
    raise EKANBadState.Create('CalibrateAlpha: ValidationCount must be > 0');

  AlphaInitial := GetCurrentAlpha;
  AlphaCenter  := AlphaInitial;
  AlphaEMA     := AlphaInitial;
  AlphaBest    := AlphaInitial;
  LossBest     := Infinity;
  LastLayer    := GetLastLayer;

  // Snapshot coefficients up-front so the sweep's NLMS drift in every
  // inference-mode forward pass can be undone before we return. SetLength=0
  // is the "no snapshot taken" sentinel used by the cleanup path below.
  SetLength(CoeffSnapshot, 0);
  if PreserveCoeffs then
    CoeffSnapshot := SnapshotAllCoeffs;

  InputVol    := TNNetVolume.Create;
  ExpectedVol := TNNetVolume.Create;
  try
    WriteLn(Format(
      'KAN alpha calibration: %d iterations x %d samples, initial alpha=%.3f' +
      ' (preserve=%s, best-alpha=%s)',
      [Iterations, SamplesPerIter, AlphaCenter,
       BoolToStr(PreserveCoeffs, True), BoolToStr(UseBestAlpha, True)]));

    for Iter := 0 to Iterations - 1 do
    begin
      LossSumCenter := 0;
      LossSumUp     := 0;
      LossSumDown   := 0;

      AlphaUp   := AlphaCenter + AlphaStep;
      AlphaDown := AlphaCenter - AlphaStep;
      if AlphaUp   > AlphaMax then AlphaUp   := AlphaMax;
      if AlphaDown < AlphaMin then AlphaDown := AlphaMin;

      for S := 0 to SamplesPerIter - 1 do
      begin
        Idx := Random(ValidationCount);
        GetValidationPair(Idx, 0, InputVol, ExpectedVol);
        AccumulateLoss(AlphaCenter, LossSumCenter);
        AccumulateLoss(AlphaUp,     LossSumUp);
        AccumulateLoss(AlphaDown,   LossSumDown);
      end;

      // Central finite difference. Divide-by-zero guarded by AlphaStep > 0.
      Gradient := (LossSumUp - LossSumDown) / (2 * (AlphaUp - AlphaDown) * SamplesPerIter);
      NewAlpha := AlphaCenter - LearningRate * Gradient;
      if NewAlpha < AlphaMin then NewAlpha := AlphaMin;
      if NewAlpha > AlphaMax then NewAlpha := AlphaMax;

      AlphaEMA := (1 - EMABeta) * AlphaEMA + EMABeta * NewAlpha;

      MeanLoss := LossSumCenter / SamplesPerIter;
      if MeanLoss < LossBest then
      begin
        LossBest  := MeanLoss;
        AlphaBest := AlphaCenter;
      end;

      WriteLn(Format(
        '  iter %2d: alpha=%.3f loss=%.4f grad=%+.4f -> new_alpha=%.3f ema=%.3f',
        [Iter, AlphaCenter, MeanLoss, Gradient, NewAlpha, AlphaEMA]));

      AlphaCenter := NewAlpha;
    end;

    if UseBestAlpha then FinalAlpha := AlphaBest
                    else FinalAlpha := AlphaEMA;

    if PreserveCoeffs then
      RestoreAllCoeffs(CoeffSnapshot);
    SetAllAlpha(FinalAlpha);

    WriteLn(Format(
      'KAN alpha calibration done. Final alpha=%.3f (was %.3f).' +
      ' best-loss=%.4f at alpha=%.3f; ema alpha=%.3f',
      [FinalAlpha, AlphaInitial, LossBest, AlphaBest, AlphaEMA]));
  finally
    InputVol.Free;
    ExpectedVol.Free;
  end;
end;

// =====================================================================
//  CreateLayer plug-in registration
// =====================================================================

// Returns a passthrough shell for TNNetKANNormaliser when the base
// CreateLayer dispatch falls through. Used by TNNetDataParallelism /
// TNNet.Clone, which serialise the original network via
// SaveStructureToString and rehydrate the layer chain by class name.
// Cloned networks (worker threads) never have LockToInference called
// on them -- FInferenceMode stays false throughout -- so the cloned
// KAN normaliser only ever needs to behave as identity passthrough.
// We return a TNNetKANNormaliser shell (via the parameterless
// CreatePassthrough constructor) rather than a plain TNNetIdentity so
// CopyWeights' source.ClassName = dest.ClassName check is satisfied --
// otherwise the framework spams "Origin class name TNNetKANNormaliser
// differs from TNNetIdentity at copy weights" on every weight sync.
// The shell has FBasis = FRNG = nil and FInferenceMode = false, so
// Compute short-circuits to inherited (TNNetIdentity) passthrough and
// the nil shared resources are never dereferenced.
//
// Returning nil means "not a class I handle" -- the registry then tries
// other registered factories before raising the original exception.
function KANCustomLayerFactory(const strData: string): TNNetLayer;
var
  S: TStringList;
begin
  Result := nil;
  S := CreateTokenizedStringList(strData, ':');
  try
    if (S.Count > 0) and (S[0] = 'TNNetKANNormaliser') then
      Result := TNNetKANNormaliser.CreatePassthrough;
  finally
    S.Free;
  end;
end;

initialization
  RegisterCustomLayerFactory(@KANCustomLayerFactory);

end.
