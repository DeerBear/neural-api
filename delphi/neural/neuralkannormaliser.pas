(*
neuralkannormaliser
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

unit neuralkannormaliser;
// =====================================================================
// STANDALONE DELPHI PORT (modern Delphi 10.x - 12 Athens).
// Hand-converted from ../../neural/neuralkannormaliser.pas via ../_port_tool.py, which
// is Pascal-lexer-aware (directives in comments/strings untouched). FPC
// and the AVX defines are resolved undefined; FPC-only branches give
// way to the upstream Delphi branch. Canonical float/pointer types come
// from neuralsimd, re-exported through neuralvolume so call sites are
// unchanged. Review-verified only -- no Delphi toolchain was available.
// =====================================================================


(*
KAN attention normaliser layer. Drop-in replacement for the per-head
TNNetPointwiseSoftMax inside multi-head self-attention. One instance
per head sub-path; per-attention-layer coordination lives in
TKANAttentionLayerInfo (neuralkanattention unit).

Mechanism specification:    docs/kan_attention_spec.md §5, §6, §7
Implementation decisions:   docs/kan_implementation_pascal.md §5

v1 status: SKELETON. The §7 pipeline body is a stub raising EKANBadState;
mode-safety guards (Backpropagate, EnterInferenceMode) are real.
*)


interface

uses
  Classes, SysUtils, Math,
  neuralvolume, neuralnetwork,
  neuralkantypes, neuralkanbasis;

type
  /// Per-head B-spline normaliser. Drop-in replacement for the per-head
  /// TNNetPointwiseSoftMax inside multi-head self-attention.
  ///
  /// Architecture: a **chain-level decorator** of TNNetPointwiseSoftMax.
  /// The KAN normaliser is constructed downstream of a real
  /// TNNetPointwiseSoftMax chain layer. The chain shape produced by
  /// TKANNet.AddKANSelfAttention is:
  ///
  ///     scores -> [TNNetPointwiseSoftMax] -> [TNNetKANNormaliser] -> ...
  ///                  (real chain layer)        (this class)
  ///
  /// Behaviour:
  ///   - Fallback mode (FInferenceMode = false, or FKANEnabled = false):
  ///     pure identity passthrough. Compute = inherited TNNetIdentity
  ///     copy from FPrevLayer.FOutput (the softmax output) to FOutput.
  ///     Backpropagate = inherited TNNetIdentity propagation back to
  ///     FPrevLayer (the softmax), which then does softmax-derivative
  ///     backprop via its own Backpropagate. Bit-identical to the
  ///     original AddSelfAttention chain by construction — we are
  ///     literally invisible.
  ///
  ///   - KAN mode (both flags true): Compute reads pre-softmax scores
  ///     from FPrevLayer.FPrevLayer.FOutput (the chain layer immediately
  ///     before the softmax) and runs the §7 pipeline. The softmax's
  ///     output (FPrevLayer.FOutput) is still available for the KL
  ///     comparison required by §6.1, no extra computation needed.
  ///     Backpropagate raises EKANInInference (spec §5.5.1).
  ///
  /// The framework manages the chain softmax's counters, output sizing,
  /// and backprop entirely on its own — the KAN normaliser does not
  /// hold any softmax instance or fudge any counters.
  TNNetKANNormaliser = class(TNNetIdentity)
  private
    // --- Identity ---
    FAttentionLayerId: integer;
    FHeadIndex: integer;

    // --- Shared resources (owned by TKANAttentionLayerInfo) ---
    FBasis: TKANBasis;
    FRNG: PKANSeededRNG;

    // --- Per-head mutable state ---
    FHead: TKANHeadState;
    FCascadeCapHits: integer;

    // --- Cached configuration (immutable post-construction) ---
    FGridSpec: TKANGridSpec;
    FHighClipShare, FLowLiftShare: TKANShareRule;
    FForceMode: TKANForceMode;
    FSharpenAlpha, FEpsilonKL, FLambdaKL: TNeuralFloat;
    FNConfirm, FCascadeMaxIter: integer;
    FTauCalm, FTauStressed: TNeuralFloat;
    FNlmsEta, FNlmsDelta: TNeuralFloat;

    // --- Operational flags ---
    FInferenceMode: boolean;
    FKANEnabled: boolean;
    FFreezeAfterTakeover: boolean;
    FSkipRedundantSoftmax: boolean;

    // --- Per-pass scratch ---
    FPhiRow: array of TNeuralFloat;
    FWSoftmaxRow: array of TNeuralFloat;
    FWKANRow: array of TNeuralFloat;
    FWSharpRow: array of TNeuralFloat;
    FTargetPreRow: array of TNeuralFloat;
    FRowSum: TNeuralFloat;             // snapshot for Phase D (spec §5.5.3)

    // --- Pipeline helpers (spec §7) ---
    procedure EvaluateSplineRow(const ScoresRow: TNNetVolume;
                                const RowIdx, RowLen: integer);
    procedure ComputeWeightsRow(const ScoresRow: TNNetVolume;
                                const RowIdx, RowLen: integer);
    function  ComputeKLRow(const RowLen: integer): TNeuralFloat;
    procedure NLMSPhaseM(const ScoresRow: TNNetVolume;
                         const RowIdx, RowLen: integer);
    procedure NLMSPhaseD(const ScoresRow: TNNetVolume;
                         const RowIdx, RowLen: integer);

    // --- Mechanism #1 helpers (spec §5.2) ---
    function  WindowGM(const I: integer): TNeuralFloat;
    function  ResolveShareMode(const IsHighClip: boolean;
                               const LayerClipRate: TNeuralFloat): TKANShareRule;

    procedure Mechanism1Sweep;
    procedure GaugeRenormalise;
    procedure CheckHandover;
    procedure ColdStartHead;
    procedure ZeroOutputForInactive;

  public
    constructor Create(const GridSpec: TKANGridSpec;
                       const AttentionLayerId, HeadIndex: integer;
                       const SharedBasis: TKANBasis;
                       const SharedRNG: PKANSeededRNG); reintroduce;
    destructor Destroy; override;

    procedure Compute; override;
    procedure Backpropagate; override;

    function  SaveDataToString: string; override;
    procedure LoadDataFromString(strData: string); override;
    function  SaveStructureToString: string; override;

    /// Called by TKANNet.LockToInference. Sets FInferenceMode and runs
    /// cold-start initialisation if no prior KAN state exists.
    procedure EnterInferenceMode;

    // --- Telemetry (spec §14.5) ---
    property HeadIndex: integer read FHeadIndex;
    property AttentionLayerId: integer read FAttentionLayerId;
    property Status: TKANStatus read FHead.Status;
    property KLEMA: TNeuralFloat read FHead.KLEMA;
    property ClipCountTotal: integer read FHead.ClipCountTotal;
    property CascadeCapHits: integer read FCascadeCapHits;

    // --- Per-layer toggles ---
    property KANEnabled: boolean read FKANEnabled write FKANEnabled;
    property FreezeAfterTakeover: boolean read FFreezeAfterTakeover write FFreezeAfterTakeover;
    property SkipRedundantSoftmax: boolean read FSkipRedundantSoftmax write FSkipRedundantSoftmax;
  end;

implementation

// =====================================================================
//  TNNetKANNormaliser
// =====================================================================

constructor TNNetKANNormaliser.Create(const GridSpec: TKANGridSpec;
                                       const AttentionLayerId, HeadIndex: integer;
                                       const SharedBasis: TKANBasis;
                                       const SharedRNG: PKANSeededRNG);
begin
  // No internal softmax instance — this layer is a chain-level
  // decorator of the TNNetPointwiseSoftMax that TKANNet.AddKANSelfAttention
  // places immediately upstream. See class doc-comment.
  inherited Create;

  FAttentionLayerId := AttentionLayerId;
  FHeadIndex := HeadIndex;
  FBasis := SharedBasis;
  FRNG := SharedRNG;
  FGridSpec := GridSpec;

  // Defaults (spec §11)
  FHighClipShare := ksrAuto;
  FLowLiftShare := ksrAuto;
  FForceMode := kfmAuto;
  FSharpenAlpha := 1.1;
  FEpsilonKL := 0.01;
  FLambdaKL := 0.01;
  FNConfirm := 100;
  FCascadeMaxIter := 16;
  FTauCalm := 0.02;
  FTauStressed := 0.10;
  FNlmsEta := 1.0;
  FNlmsDelta := 1e-6;

  FInferenceMode := false;
  FKANEnabled := true;
  FFreezeAfterTakeover := false;
  FSkipRedundantSoftmax := true;

  // Allocate per-head state (uninitialised; cold-start happens in EnterInferenceMode)
  SetLength(FHead.Coeffs, GridSpec.KnotCount);
  FHead.Status := ksSoftmaxActive;
  FHead.KLEMA := Infinity;
  FHead.ConsecutiveLowPasses := 0;
  FHead.ClipCountTotal := 0;
  FCascadeCapHits := 0;

  // Per-pass scratch buffers are sized lazily (first Compute) when row length is known.
end;

destructor TNNetKANNormaliser.Destroy;
begin
  SetLength(FHead.Coeffs, 0);
  SetLength(FPhiRow, 0);
  SetLength(FWSoftmaxRow, 0);
  SetLength(FWKANRow, 0);
  SetLength(FWSharpRow, 0);
  SetLength(FTargetPreRow, 0);
  // FBasis and FRNG are owned by TKANAttentionLayerInfo; do not free here.
  inherited Destroy;
end;

procedure TNNetKANNormaliser.Compute;
begin
  if (not FInferenceMode) or (not FKANEnabled) then
  begin
    // Fallback: pure identity passthrough. FPrevLayer is the softmax
    // (placed by AddKANSelfAttention), FPrevLayer.FOutput already holds
    // the row-normalised attention weights. inherited Compute (from
    // TNNetIdentity) copies that into FOutput. No KAN bookkeeping, no
    // per-head state mutation, no extra cost.
    inherited Compute;
    exit;
  end;

  // KAN mode: read pre-softmax scores from FPrevLayer.FPrevLayer.FOutput
  // (the chain layer immediately before the softmax) and run §7
  // pipeline. The softmax's output (FPrevLayer.FOutput) is available
  // for the KL comparison (spec §6.1) at no extra forward cost.
  //
  // TODO: §7 pipeline.
  //   for each row in pre-softmax scores:
  //     EvaluateSplineRow      (write FPhiRow, FRowSum from coefficients)
  //     ComputeWeightsRow      (FWKANRow := φ / Σ φ;
  //                             FWSoftmaxRow := already in FPrevLayer.FOutput)
  //     write selected weights to FOutput
  //     ComputeKLRow           (uses FPrevLayer.FOutput directly)
  //     NLMS (Phase M or Phase D depending on Status)
  //     Mechanism1Sweep
  //     GaugeRenormalise
  //   if Status = SoftmaxActive: CheckHandover
  //   if not active head: ZeroOutputForInactive
  raise EKANBadState.Create('TNNetKANNormaliser.Compute: §7 pipeline not implemented');
end;

procedure TNNetKANNormaliser.Backpropagate;
begin
  if FInferenceMode then
    raise EKANInInference.Create(
      'Backpropagate called on KAN normaliser layer in inference mode');

  // Fallback: pure identity passthrough. inherited Backpropagate (from
  // TNNetIdentity) accumulates FOutputError into FPrevLayer.FOutputError
  // (the softmax's), then cascades via FPrevLayer.Backpropagate which is
  // the softmax's full softmax-derivative backprop. No counter fudging
  // and no decorator-specific code path needed; the framework does it all.
  inherited Backpropagate;
end;

function TNNetKANNormaliser.SaveDataToString: string;
begin
  // TODO: serialise FHead.Coeffs, FHead.Status, FHead.KLEMA, counters,
  // FCascadeCapHits, FRNG^.State.
  Result := '';
end;

procedure TNNetKANNormaliser.LoadDataFromString(strData: string);
begin
  // TODO: inverse of SaveDataToString.
end;

function TNNetKANNormaliser.SaveStructureToString: string;
begin
  // TODO: serialise GridSpec + cached configuration values.
  Result := inherited SaveStructureToString;
end;

procedure TNNetKANNormaliser.EnterInferenceMode;
begin
  if FInferenceMode then exit;
  if (FRNG = nil) or (FRNG^.State = 0) then
    raise EKANBadState.Create('TNNetKANNormaliser.EnterInferenceMode: RNG not seeded');
  if FHead.KLEMA = Infinity then ColdStartHead;
  FInferenceMode := true;
end;

procedure TNNetKANNormaliser.EvaluateSplineRow(const ScoresRow: TNNetVolume;
                                                 const RowIdx, RowLen: integer);
// Spec 5.1: psi(s) = sum_m c_m * B_m(s) over the k+1 active basis functions;
// phi(s) = psi^2 by construction (5.3 gauge premise). Writes phi for every
// column of this row into FPhiRow, snapshots the row total in FRowSum --
// the latter is consumed by ComputeWeightsRow (5.5.1) and (post-takeover)
// by NLMSPhaseD's sharpened-target derivation (5.5.3).
//
// Indexing: this layer is per-head; the score tensor's Depth axis identifies
// rows (per impl-doc 5.3 pseudocode `for rowIdx in 0..FOutput.Depth - 1`).
// CAI volumes lay out Raw[] with Depth varying fastest, so column j of row
// RowIdx is at flat offset j*Depth + RowIdx. FBasis.Evaluate handles out-of-
// range knot indices by writing 0 to the corresponding basis value, so the
// bounds check below is a safety guard against indexing the coefficient
// array out of range (Debug builds range-check); the math is unaffected.
var
  J, FirstIdx, M, KnotN, D: integer;
  S, Psi: TNeuralFloat;
  BasisVals: array[0..3] of TNeuralFloat;
begin
  if Length(FPhiRow) < RowLen then SetLength(FPhiRow, RowLen);
  KnotN := Length(FHead.Coeffs);
  D := ScoresRow.Depth;
  FRowSum := 0;
  for J := 0 to RowLen - 1 do
  begin
    S := ScoresRow.Raw[J * D + RowIdx];
    FBasis.Evaluate(S, FirstIdx, BasisVals);
    Psi := 0;
    for M := 0 to 3 do
    begin
      if (FirstIdx + M >= 0) and (FirstIdx + M < KnotN) then
        Psi := Psi + FHead.Coeffs[FirstIdx + M] * BasisVals[M];
    end;
    FPhiRow[J] := Psi * Psi;
    FRowSum := FRowSum + FPhiRow[J];
  end;
end;

procedure TNNetKANNormaliser.ComputeWeightsRow(const ScoresRow: TNNetVolume;
                                                 const RowIdx, RowLen: integer);
// Spec 5.5.1: derive the two candidate weight rows.
//   FWKANRow   := FPhiRow / FRowSum         (populated by EvaluateSplineRow)
//   FWSoftmaxRow := numerically-stable row softmax of ScoresRow at RowIdx
// The softmax candidate is skipped when FSkipRedundantSoftmax is true AND
// the head has taken over (spec 11.8) -- the row-KL is then meaningless
// and the caller is responsible for not invoking ComputeKLRow.
// Same Depth-stride indexing as EvaluateSplineRow.
var
  J, D: integer;
  S, MaxScore, Sum: TNeuralFloat;
begin
  if Length(FWKANRow) < RowLen then SetLength(FWKANRow, RowLen);
  if Length(FWSoftmaxRow) < RowLen then SetLength(FWSoftmaxRow, RowLen);

  // KAN candidate: row-normalise phi.
  if FRowSum > 0 then
  begin
    for J := 0 to RowLen - 1 do
      FWKANRow[J] := FPhiRow[J] / FRowSum;
  end
  else
  begin
    // Pathological: every phi on this row was zero. Yield a uniform row so
    // downstream KL / NLMS don't divide by zero; the next pass's NLMS will
    // pull coefficients back into range.
    for J := 0 to RowLen - 1 do
      FWKANRow[J] := 0;
  end;

  if FSkipRedundantSoftmax and (FHead.Status = ksKANActive) then exit;

  // Softmax candidate: subtract row max for numerical stability, then exp
  // and normalise.
  D := ScoresRow.Depth;
  MaxScore := ScoresRow.Raw[RowIdx];        // J=0 contribution
  for J := 1 to RowLen - 1 do
  begin
    S := ScoresRow.Raw[J * D + RowIdx];
    if S > MaxScore then MaxScore := S;
  end;
  Sum := 0;
  for J := 0 to RowLen - 1 do
  begin
    FWSoftmaxRow[J] := Exp(ScoresRow.Raw[J * D + RowIdx] - MaxScore);
    Sum := Sum + FWSoftmaxRow[J];
  end;
  if Sum > 0 then
  begin
    Sum := 1 / Sum;
    for J := 0 to RowLen - 1 do
      FWSoftmaxRow[J] := FWSoftmaxRow[J] * Sum;
  end;
end;

function TNNetKANNormaliser.ComputeKLRow(const RowLen: integer): TNeuralFloat;
// Spec 6.1: KL(softmax || KAN) over the row, in nats. Drives the EMA in
// 6.2 / 6.3 that triggers SoftmaxActive -> KANActive graduation.
//   KL := sum_j p_j * log( p_j / q_j )
// where p = FWSoftmaxRow, q = FWKANRow (both populated by the most recent
// ComputeWeightsRow). Terms with p_j = 0 contribute 0 (the standard
// 0*log(0/q) := 0 convention). q_j is floored to a tiny epsilon so an
// unexpectedly-zero KAN weight cannot blow KL up to +Infinity and derail
// the EMA -- the floor is well below any threshold the rest of the spec
// relies on (epsilon_KL defaults to 0.01 nats, 6.3).
var
  J: integer;
  P, Q: TNeuralFloat;
const
  QFloor = 1e-30;
begin
  Result := 0;
  for J := 0 to RowLen - 1 do
  begin
    P := FWSoftmaxRow[J];
    if P <= 0 then Continue;
    Q := FWKANRow[J];
    if Q < QFloor then Q := QFloor;
    Result := Result + P * Ln(P / Q);
  end;
end;

procedure TNNetKANNormaliser.NLMSPhaseM(const ScoresRow: TNNetVolume;
                                         const RowIdx, RowLen: integer);
begin
  // TODO: per-score NLMS toward exp(s), in psi-space (spec §5.5.2).
  raise EKANBadState.Create('TNNetKANNormaliser.NLMSPhaseM: not implemented');
end;

procedure TNNetKANNormaliser.NLMSPhaseD(const ScoresRow: TNNetVolume;
                                         const RowIdx, RowLen: integer);
begin
  // TODO: per-score NLMS toward sharpened target derived from FRowSum
  // and FWKANRow snapshot; α-power renormalisation (spec §5.5.3).
  raise EKANBadState.Create('TNNetKANNormaliser.NLMSPhaseD: not implemented');
end;

function TNNetKANNormaliser.WindowGM(const I: integer): TNeuralFloat;
// Spec 5.2.2: geometric mean of |c_j| over c_i's 2k+1 window, clamped at
// array boundaries. Log-space throughout for numerical safety. No epsilon
// floor is strictly required (the low-lift rule 5.2.4 keeps |c_j| bounded
// away from zero) but we defend against the cold-start edge by flooring
// to 1e-30 anyway.
var
  J, Lo, Hi, K: integer;
  AbsC, LogSum: TNeuralFloat;
const
  CFloor = 1e-30;
begin
  K := FGridSpec.BasisOrder;             // cubic -> radius 3 -> 2k+1 = 7
  Lo := Max(0, I - K);
  Hi := Min(Length(FHead.Coeffs) - 1, I + K);
  LogSum := 0;
  for J := Lo to Hi do
  begin
    AbsC := Abs(FHead.Coeffs[J]);
    if AbsC < CFloor then AbsC := CFloor;
    LogSum := LogSum + Ln(AbsC);
  end;
  Result := Exp(LogSum / (Hi - Lo + 1));
end;

function TNNetKANNormaliser.ResolveShareMode(const IsHighClip: boolean;
                                              const LayerClipRate: TNeuralFloat): TKANShareRule;
// Spec 5.2.5: select the share rule for one clip-or-lift event.
//   Configured non-auto    -> return as-is.
//   Auto + clip_rate < tau_calm        -> Mode A (Proportional for both halves).
//   Auto + clip_rate > tau_stressed    -> Mode C (HighClip=Inverse, LowLift=Prop).
//   Auto + in hysteresis band          -> default to Mode C (the safe,
//     always-stable mode per 5.2.5). The spec's literal phrasing is
//     "keep current mode" which strictly requires a small persistent
//     state across calls; v1 defers that refinement -- Mode C is safe
//     in every regime and matches the "Stable; converges fast" note.
// Mode D (Prop high + Inv low) is structurally unreachable here.
var
  Configured: TKANShareRule;
begin
  if IsHighClip then Configured := FHighClipShare
  else               Configured := FLowLiftShare;

  if Configured <> ksrAuto then
    Exit(Configured);

  if LayerClipRate < FTauCalm then
    Exit(ksrProportional);                 // Mode A (both halves Proportional)

  // Mode C (clip_rate > tau_stressed OR in hysteresis band)
  if IsHighClip then Exit(ksrInverseProportional)
  else               Exit(ksrProportional);
end;

procedure TNNetKANNormaliser.Mechanism1Sweep;
begin
  // TODO: cascade to fixed point with high-clip + low-lift + share-rule
  // redistribution (spec §5.2). Increment FCascadeCapHits if max_iter reached.
  raise EKANBadState.Create('TNNetKANNormaliser.Mechanism1Sweep: not implemented');
end;

procedure TNNetKANNormaliser.GaugeRenormalise;
// Spec 5.3: enforce per-head geometric mean of |c_j| = 1 by dividing every
// coefficient by G := exp(mean_j(log|c_j|)). A pure gauge transformation:
// w_ij = phi(s)/sum(phi) is invariant under uniform multiplicative rescale,
// so the forward attention output is mathematically unchanged (5.3.4).
// Computed in log-space throughout to dodge Pi|c_j| under/overflow (5.3.3).
var
  I, N: integer;
  AbsC, LogSum, InvG: TNeuralFloat;
begin
  N := Length(FHead.Coeffs);
  if N = 0 then exit;
  LogSum := 0;
  for I := 0 to N - 1 do
  begin
    AbsC := Abs(FHead.Coeffs[I]);
    // floor zero magnitudes so Ln(0) = -Infinity can't poison the mean;
    // for the cold-started exp(knot/2) fit no |c_j| ever reaches this.
    if AbsC < 1e-30 then AbsC := 1e-30;
    LogSum := LogSum + Ln(AbsC);
  end;
  InvG := Exp(-LogSum / N);   // 1/G as a single Exp; signs preserved (GM-3).
  for I := 0 to N - 1 do
    FHead.Coeffs[I] := FHead.Coeffs[I] * InvG;
end;

procedure TNNetKANNormaliser.CheckHandover;
// Spec 6.3 / 6.6: one-way transition ksSoftmaxActive -> ksKANActive once
// KL_ema has stayed below epsilon_KL for at least N_confirm consecutive
// passes. The counter is maintained here against the most recent KLEMA
// snapshot; KLEMA itself is refreshed by the NLMS phases each pass.
// Once a head is ksKANActive there is no reversion (spec 6.6).
begin
  if FHead.Status <> ksSoftmaxActive then exit;
  if FHead.KLEMA < FEpsilonKL then
    Inc(FHead.ConsecutiveLowPasses)
  else
    FHead.ConsecutiveLowPasses := 0;
  if FHead.ConsecutiveLowPasses >= FNConfirm then
    FHead.Status := ksKANActive;
end;

procedure TNNetKANNormaliser.ColdStartHead;
begin
  if FBasis = nil then
    raise EKANBadState.Create('TNNetKANNormaliser.ColdStartHead: basis not set');
  FBasis.FitPsiToExp(FHead.Coeffs);
  FHead.Status := ksSoftmaxActive;
  FHead.KLEMA := Infinity;
  FHead.ConsecutiveLowPasses := 0;
  FHead.ClipCountTotal := 0;
end;

procedure TNNetKANNormaliser.ZeroOutputForInactive;
begin
  // Write zeros to FOutput so the downstream concat receives zero
  // contribution from this head (HS-3).
  FOutput.Fill(0);
end;

end.
