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

{$include neuralnetwork.inc}

interface

uses
  Classes, SysUtils, Math,
  neuralvolume, neuralnetwork,
  neuralkantypes, neuralkanbasis;

type
  /// Per-head B-spline normaliser. Drop-in replacement for the per-head
  /// TNNetPointwiseSoftMax inside multi-head self-attention.
  ///
  /// Architecture: this is a **decorator** of TNNetPointwiseSoftMax, not
  /// a subclass. It inherits TNNetIdentity (matching the impl-doc §5.1
  /// "drop-in replacement, not extension" framing) and holds an internal
  /// TNNetPointwiseSoftMax instance to which it delegates in fallback
  /// mode. Concrete behaviour:
  ///
  ///   - In training mode (FInferenceMode = false), or when KAN is
  ///     disabled (FKANEnabled = false), Compute and Backpropagate
  ///     delegate to the decorated softmax instance — bit-identical
  ///     to TNNetPointwiseSoftMax with no risk of behavioural drift.
  ///   - In inference mode with KANEnabled = true, Compute runs the
  ///     §7 KAN pipeline. Backpropagate raises EKANInInference (the
  ///     spec forbids backprop while KAN is active; §5.5.1).
  TNNetKANNormaliser = class(TNNetIdentity)
  private
    // --- Identity ---
    FAttentionLayerId: integer;
    FHeadIndex: integer;

    // --- Shared resources (owned by TKANAttentionLayerInfo) ---
    FBasis: TKANBasis;
    FRNG: PKANSeededRNG;

    // --- Decorated softmax instance ---
    // We hold a literal TNNetPointwiseSoftMax instance and delegate to it
    // in fallback mode. This is a pure decorator: the softmax is
    // configured with our FPrevLayer (so it reads the same input we read)
    // and its FOutput is copied into ours after delegation. The softmax
    // is owned by us; it is NOT part of any TNNet's layer chain.
    FInternalSoftmax: TNNetPointwiseSoftMax;

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
    procedure Mechanism1Sweep;
    procedure GaugeRenormalise;
    procedure CheckHandover;
    procedure ColdStartHead;
    procedure ZeroOutputForInactive;

    // --- Soft-mode forward (used when InferenceMode=false or KANEnabled=false) ---
    procedure ComputeAsSoftmax;

  public
    constructor Create(const GridSpec: TKANGridSpec;
                       const AttentionLayerId, HeadIndex: integer;
                       const SharedBasis: TKANBasis;
                       const SharedRNG: PKANSeededRNG); reintroduce;
    destructor Destroy; override;

    procedure Compute; override;
    procedure Backpropagate; override;
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;

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
  inherited Create;

  // Decorated softmax: a literal TNNetPointwiseSoftMax instance we
  // delegate to in fallback mode. SkipBackpropDerivative=0, NoForward=0
  // are the default behaviours used by AddSelfAttention.
  FInternalSoftmax := TNNetPointwiseSoftMax.Create(0, 0);
  // The decorated softmax is held privately; it is not part of any
  // network's chain. Set its FDepartingBranchesCnt := 1 so that when
  // we delegate to its Backpropagate, the framework's branch-barrier
  // assertion (TestBackPropCallCurrCnt) sees a self-consistent state.
  FInternalSoftmax.FDepartingBranchesCnt := 1;

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
  // FInternalSoftmax is owned by us — free it before the parent runs.
  FreeAndNil(FInternalSoftmax);
  SetLength(FHead.Coeffs, 0);
  SetLength(FPhiRow, 0);
  SetLength(FWSoftmaxRow, 0);
  SetLength(FWKANRow, 0);
  SetLength(FWSharpRow, 0);
  SetLength(FTargetPreRow, 0);
  // FBasis and FRNG are owned by TKANAttentionLayerInfo; do not free here.
  inherited Destroy;
end;

procedure TNNetKANNormaliser.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  // Wire the decorated softmax to the same input we read. Its FOutput
  // and FOutputError get sized by its own SetPrevLayer's resize chain
  // to match the input — exactly what TNNetPointwiseSoftMax does when
  // it's added to a network normally.
  FInternalSoftmax.SetPrevLayer(pPrevLayer);
end;

procedure TNNetKANNormaliser.Compute;
begin
  if (not FInferenceMode) or (not FKANEnabled) then
  begin
    ComputeAsSoftmax;
    exit;
  end;

  // TODO: §7 pipeline.
  //   for each row in input:
  //     EvaluateSplineRow
  //     ComputeWeightsRow
  //     write selected weights to FOutput
  //     update KL EMA
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

  // Standard branch-tracking for our own counter (the framework expects
  // every layer's Backpropagate to manage these counters).
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  TestBackPropCallCurrCnt;

  // Delegate to the decorated softmax. It performs:
  //   - the softmax derivative computation (x*(1-x))
  //   - accumulation into FPrevLayer.OutputError
  //   - cascade via FPrevLayer.Backpropagate
  // The framework does not reset FInternalSoftmax's counter (since the
  // softmax is held privately, not registered in any chain), so we
  // reset it ourselves before each delegation.
  FInternalSoftmax.FBackPropCallCurrentCnt := 0;
  FInternalSoftmax.FOutputError.CopyNoChecks(FOutputError);
  FInternalSoftmax.Backpropagate;

  // We do NOT call inherited Backpropagate here — that would also
  // accumulate FOutputError into FPrevLayer.OutputError, double-counting
  // the gradient.
end;

procedure TNNetKANNormaliser.ComputeAsSoftmax;
begin
  // Pure decorator: delegate to the held softmax instance.
  //   1. The softmax reads FPrevLayer.FOutput (same as ours, wired in
  //      SetPrevLayer) and writes to FInternalSoftmax.FOutput.
  //   2. We copy that into our own FOutput so downstream layers see it.
  //
  // Used:
  //   - during training (FInferenceMode = false), so the layer behaves
  //     identically to the original softmax it replaces;
  //   - when this layer's KANEnabled is false (selective-deployment
  //     opt-out, spec §11.8 / impl doc §4.6).
  FInternalSoftmax.Compute;
  FOutput.CopyNoChecks(FInternalSoftmax.FOutput);
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
begin
  // TODO: for each column j in row RowIdx of ScoresRow,
  //   ψ(s) := Σ_m c_m · B_m(s)  (k+1 active basis functions)
  //   φ(s) := ψ(s)²
  //   FPhiRow[j] := φ(s)
  // Then FRowSum := Σ_j FPhiRow[j]   -- snapshot for Phase D (spec §5.5.3)
  raise EKANBadState.Create('TNNetKANNormaliser.EvaluateSplineRow: not implemented');
end;

procedure TNNetKANNormaliser.ComputeWeightsRow(const ScoresRow: TNNetVolume;
                                                 const RowIdx, RowLen: integer);
begin
  // TODO:
  //   FWKANRow[j] := FPhiRow[j] / FRowSum
  //   if not (FSkipRedundantSoftmax and Status=KANActive):
  //     FWSoftmaxRow := row_softmax(ScoresRow[RowIdx])
  raise EKANBadState.Create('TNNetKANNormaliser.ComputeWeightsRow: not implemented');
end;

function TNNetKANNormaliser.ComputeKLRow(const RowLen: integer): TNeuralFloat;
begin
  // TODO: KL(softmax || KAN) over RowLen positions, in nats.
  Result := 0;
  raise EKANBadState.Create('TNNetKANNormaliser.ComputeKLRow: not implemented');
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

procedure TNNetKANNormaliser.Mechanism1Sweep;
begin
  // TODO: cascade to fixed point with high-clip + low-lift + share-rule
  // redistribution (spec §5.2). Increment FCascadeCapHits if max_iter reached.
  raise EKANBadState.Create('TNNetKANNormaliser.Mechanism1Sweep: not implemented');
end;

procedure TNNetKANNormaliser.GaugeRenormalise;
begin
  // TODO: log-space mean of |c_j|, divide all coefficients by exp(mean).
  raise EKANBadState.Create('TNNetKANNormaliser.GaugeRenormalise: not implemented');
end;

procedure TNNetKANNormaliser.CheckHandover;
begin
  // TODO: if KLEMA < EpsilonKL for >= NConfirm consecutive passes,
  // atomically flip Status := ksKANActive (spec §6.3).
  raise EKANBadState.Create('TNNetKANNormaliser.CheckHandover: not implemented');
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
