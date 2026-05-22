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
  neuralvolume, neuralnetwork,
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

    // --- Training-method overrides that gate on lock state ---
    procedure Backpropagate(pInput: TNNetVolume); override;

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
  {$PUSH}{$Q-}{$R-}
  Seed := (UInt64(AttentionLayerId + 1) * UInt64($9E3779B97F4A7C15)) xor Spec.Hash;
  {$POP}
  if Seed = 0 then Seed := 1;

  Info := TKANAttentionLayerInfo.Create(AttentionLayerId, Spec, HMax, Seed,
    SquaringClipRate, SquaringSweeps, 0.01, InitialHeads);
  FAttentionLayers.Add(Info);

  // SharpenAlpha / KLThreshold / KLConfirmPasses are not yet propagated to
  // each TNNetKANNormaliser instance — the normaliser still uses its own
  // defaults from its constructor. Wiring these through requires adding
  // published-property setters on the normaliser; out of scope for v1.

  // (3) Q/K/V projections — mirror AddSelfAttention's no-HasNorm branch.
  QueryGroup := AddLayerAfter([TNNetPointwiseConvLinear.Create(PreviousDepth, 1)], PreviousLayer);
  KeyGroup   := AddLayerAfter([TNNetPointwiseConvLinear.Create(PreviousDepth, 1)], PreviousLayer);
  ValueGroup := AddLayerAfter([TNNetPointwiseConvLinear.Create(PreviousDepth, 1)], PreviousLayer);
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
begin
  // TODO: dump (layer_idx, ActiveHeads, status_distribution, KL_ema_avg,
  // clip_rate_ema, cascade_cap_hits) per attention layer.
  Result := '';
end;

procedure TKANNet.Backpropagate(pInput: TNNetVolume);
begin
  AssertNotLocked('Backpropagate');
  inherited Backpropagate(pInput);
end;

end.
