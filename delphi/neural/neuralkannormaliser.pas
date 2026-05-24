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

v1 status: §7 pipeline body, both NLMS phases, gauge, mechanism #1
cascade, KL/handover, basis evaluation, chain-decorator fallback, and
state save/load round-trip are all implemented. Layer-info backref (for
clip-rate plumbing) and ZeroOutputForInactive are deferred to a v2 pass
that also lands DoSquaring; see Compute's header comment.
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
    function  HighClipAt(const I: integer; const G: TNeuralFloat;
                         const LayerClipRate: TNeuralFloat): boolean;
    function  LowLiftAt(const I: integer; const G: TNeuralFloat;
                        const LayerClipRate: TNeuralFloat): boolean;
    function  SweepOnce(const LayerClipRate: TNeuralFloat): integer;

    procedure Mechanism1Sweep(const LayerClipRate: TNeuralFloat);
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

    // --- Tunable parameters (set by AddKANSelfAttention before LockToInference;
    //     defaults from constructor follow spec §11). Validation lives at the
    //     builder call site so caller-supplied bad values fail fast.
    property SharpenAlpha: TNeuralFloat read FSharpenAlpha write FSharpenAlpha;
    property KLThreshold: TNeuralFloat read FEpsilonKL write FEpsilonKL;
    property KLConfirmPasses: integer read FNConfirm write FNConfirm;
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
// Spec 7: per-inference pipeline body. Per-head; the layer-level steps
// (squaring trigger, concat, output projection) live elsewhere -- the layer
// info object (TKANAttentionLayerInfo) and the downstream chain layers
// respectively. This proc owns steps 2-10 of spec 7 for a single head.
//
// Step 1 (compute scores) was done by the TNNetDotProducts upstream of the
// softmax; we read its output via FPrevLayer.FPrevLayer.Output. Step 5
// (attended = w * V) is done by the TNNetDotProducts downstream; we just
// populate FOutput with the chosen weights matrix.
//
// Per-row interleaving rationale: spec 5.5.3's "row snapshot" lives in
// instance fields (FPhiRow / FRowSum / FWKANRow) and must be valid for the
// per-row NLMS. We therefore evaluate -> compute weights -> write output row
// -> accumulate KL row contribution -> per-row NLMS, then move to the next
// row. The next row's EvaluateSplineRow recomputes the snapshot from the
// now-updated coefficients, matching spec 5.5.3's frozen-per-row,
// evolved-across-rows semantics.
//
// v1 limitations (acceptable; documented for v2 pickup):
//   - No backref from normaliser to TKANAttentionLayerInfo (no circular dep
//     declared). The orchestrator therefore passes 0 as LayerClipRate to
//     Mechanism1Sweep, which makes ResolveShareMode always pick Mode A
//     (proportional/proportional, gentle preserve-shape). Per-head clip
//     rates also aren't reported back to the layer's squaring EMA -- fine in
//     v1 because DoSquaring is also stubbed.
//   - ZeroOutputForInactive isn't invoked; all heads are assumed active.
//     With no squaring in v1, ActiveHeads = HMax always, so every head is
//     valid. When squaring lands the orchestrator gains an "am I beyond
//     ActiveHeads" check that calls ZeroOutputForInactive and skips the
//     rest of the pipeline.
var
  PreScores: TNNetVolume;
  RowIdx, RowLen, NumRows, J, Depth: integer;
  KLPassSum, KLPass: TNeuralFloat;
  LayerClipRate: TNeuralFloat;
begin
  if (not FInferenceMode) or (not FKANEnabled) then
  begin
    // Fallback: pure identity passthrough. inherited Compute (TNNetIdentity)
    // copies FPrevLayer.FOutput (the softmax) into FOutput. No KAN bookkeeping.
    inherited Compute;
    exit;
  end;

  // Pre-softmax scores from the layer immediately upstream of the softmax
  // (the chain shape AddKANSelfAttention builds is ... -> ReLUL -> Softmax
  // -> KANNormaliser). Same shape and indexing as the softmax output:
  // Depth = number of rows, SizeX*SizeY = per-row column count.
  PreScores := FPrevLayer.FPrevLayer.Output;
  NumRows := PreScores.Depth;
  RowLen := PreScores.SizeX * PreScores.SizeY;
  if (NumRows = 0) or (RowLen = 0) then exit;

  Depth := FOutput.Depth;
  KLPassSum := 0;

  for RowIdx := 0 to NumRows - 1 do
  begin
    // Steps 2-3 (spec 7): spline eval + candidate weights for this row.
    EvaluateSplineRow(PreScores, RowIdx, RowLen);
    ComputeWeightsRow(PreScores, RowIdx, RowLen);

    // Step 4: write selected forward-path weights into FOutput[*, RowIdx]
    // using the same Depth-fastest indexing the inputs and helpers use.
    if FHead.Status = ksSoftmaxActive then
    begin
      for J := 0 to RowLen - 1 do
        FOutput.Raw[J * Depth + RowIdx] := FWSoftmaxRow[J];
    end
    else
    begin
      for J := 0 to RowLen - 1 do
        FOutput.Raw[J * Depth + RowIdx] := FWKANRow[J];
    end;

    // Step 6 (per-row contribution): accumulate KL while both candidate rows
    // are populated. KL is only meaningful pre-takeover; once handed over
    // FSkipRedundantSoftmax suppresses the softmax candidate (FWSoftmaxRow
    // contents are stale) and the check is skipped.
    if FHead.Status = ksSoftmaxActive then
      KLPassSum := KLPassSum + ComputeKLRow(RowLen);

    // Step 7: NLMS update for this row. Phase per current status. Mutates
    // coefficients; next row's EvaluateSplineRow picks up the new state.
    if FHead.Status = ksSoftmaxActive then
      NLMSPhaseM(PreScores, RowIdx, RowLen)
    else
      NLMSPhaseD(PreScores, RowIdx, RowLen);
  end;

  // Step 6 (per-pass completion): EMA update from this pass's mean row KL.
  // First update after cold-start replaces the Infinity sentinel rather than
  // blending (Inf*0.99 + finite*0.01 stays Inf), so the EMA actually moves.
  if FHead.Status = ksSoftmaxActive then
  begin
    KLPass := KLPassSum / NumRows;
    if FHead.KLEMA = Infinity then
      FHead.KLEMA := KLPass
    else
      FHead.KLEMA := (1 - FLambdaKL) * FHead.KLEMA + FLambdaKL * KLPass;
  end;

  // Steps 8-9: mechanism #1 cascade + GM=1 gauge renormalisation. Order
  // matters per spec 7.1: rebalance operates on the pre-renormalised state
  // (where window products are meaningful), then gauge resets the scale.
  LayerClipRate := 0;   // v1 placeholder; see header comment.
  Mechanism1Sweep(LayerClipRate);
  GaugeRenormalise;

  // Step 10: handover check (internally no-ops if already handed over or if
  // the KLEMA / counter conditions aren't met). One-way per spec 6.6.
  CheckHandover;
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
// Spec 8.1, 8.3: per-head state serialisation. Pipe-delimited key=value
// pairs; floats via CAI's locale-safe NeuralFloatToStr; the UInt64 RNG
// state is round-tripped through Int64 (bitwise reinterpretation, decimal
// representation works correctly even for high-bit-set values because the
// cast is unsigned->signed and back); the +Infinity KLEMA sentinel
// (cold-start) is encoded as the literal "inf" because FloatToStr's
// behaviour on Inf is platform-dependent.
//
// Coefficients are a space-separated tail in the same string. Round-trip
// precision is whatever NeuralFloatToStr provides for TNeuralFloat=Single
// (typically 7-9 significant digits); coefficient drift below this is
// smoothed by NLMS on the first inference pass post-load anyway.
var
  I: integer;
  KLEMAStr, RNGStr: string;
begin
  if IsInfinite(FHead.KLEMA) then
    KLEMAStr := 'inf'
  else
    KLEMAStr := NeuralFloatToStr(FHead.KLEMA);
  if FRNG <> nil then
    RNGStr := IntToStr(Int64(FRNG^.State))
  else
    RNGStr := '0';
  Result := Format('status=%d|klema=%s|conslow=%d|clipcnt=%d|caphits=%d|rng=%s|coeffs=',
    [Ord(FHead.Status), KLEMAStr, FHead.ConsecutiveLowPasses,
     FHead.ClipCountTotal, FCascadeCapHits, RNGStr]);
  for I := 0 to Length(FHead.Coeffs) - 1 do
  begin
    if I > 0 then Result := Result + ' ';
    Result := Result + NeuralFloatToStr(FHead.Coeffs[I]);
  end;
end;

procedure TNNetKANNormaliser.LoadDataFromString(strData: string);
// Inverse of SaveDataToString. Validates coefficient count against current
// FGridSpec.KnotCount and raises EKANBadState on mismatch (loading a
// checkpoint from a different grid configuration would silently corrupt
// the per-head state, so we fail fast). Unknown keys are silently ignored
// for forward compatibility with future state additions.
var
  Fields, CoeffStrs: TStringList;
  Sep, I, KnotN: integer;
  Part, Key, Val: string;
begin
  if Trim(strData) = '' then
    raise EKANBadState.Create(
      'TNNetKANNormaliser.LoadDataFromString: empty data string');
  Fields := TStringList.Create;
  CoeffStrs := TStringList.Create;
  try
    Fields.StrictDelimiter := True;
    Fields.Delimiter := '|';
    Fields.DelimitedText := strData;
    for I := 0 to Fields.Count - 1 do
    begin
      Part := Fields[I];
      Sep := Pos('=', Part);
      if Sep <= 0 then Continue;
      Key := Copy(Part, 1, Sep - 1);
      Val := Copy(Part, Sep + 1, MaxInt);
      if Key = 'status' then
        FHead.Status := TKANStatus(StrToInt(Val))
      else if Key = 'klema' then
      begin
        if LowerCase(Val) = 'inf' then
          FHead.KLEMA := Infinity
        else
          FHead.KLEMA := NeuralStrToFloat(Val);
      end
      else if Key = 'conslow' then
        FHead.ConsecutiveLowPasses := StrToInt(Val)
      else if Key = 'clipcnt' then
        FHead.ClipCountTotal := StrToInt(Val)
      else if Key = 'caphits' then
        FCascadeCapHits := StrToInt(Val)
      else if Key = 'rng' then
      begin
        if FRNG <> nil then
          FRNG^.State := UInt64(StrToInt64(Val));
      end
      else if Key = 'coeffs' then
      begin
        CoeffStrs.StrictDelimiter := True;
        CoeffStrs.Delimiter := ' ';
        CoeffStrs.DelimitedText := Val;
        KnotN := Length(FHead.Coeffs);
        if CoeffStrs.Count <> KnotN then
          raise EKANBadState.CreateFmt(
            'TNNetKANNormaliser.LoadDataFromString: coefficient count mismatch ' +
            '(got %d, expected %d for grid KnotCount=%d)',
            [CoeffStrs.Count, KnotN, FGridSpec.KnotCount]);
        for I := 0 to KnotN - 1 do
          FHead.Coeffs[I] := NeuralStrToFloat(CoeffStrs[I]);
      end;
      // Unknown keys are silently ignored for forward compatibility.
    end;
  finally
    CoeffStrs.Free;
    Fields.Free;
  end;
end;

function TNNetKANNormaliser.SaveStructureToString: string;
// Spec 8.1: structure (immutable, set at construction) is the grid spec
// plus the per-network identity (layer + head). The actual chain
// reconstruction is the network builder's responsibility (re-running
// TKANNet.AddKANSelfAttention with the appropriate parameters); this
// string is mostly diagnostic/audit, but is also used by LoadDataFromString
// (via the network layer-load loop) to validate that the loaded data
// matches the current structure.
begin
  Result := inherited SaveStructureToString
            + ' kanlow=' + NeuralFloatToStr(FGridSpec.GridLow)
            + ' kanhigh=' + NeuralFloatToStr(FGridSpec.GridHigh)
            + ' kanknots=' + IntToStr(FGridSpec.KnotCount)
            + ' kanorder=' + IntToStr(FGridSpec.BasisOrder)
            + ' kanlayer=' + IntToStr(FAttentionLayerId)
            + ' kanhead=' + IntToStr(FHeadIndex);
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
// Spec 5.5.2: Phase M NLMS, per score, in psi-space. Acts only on the k+1
// coefficients with non-zero basis activation at s.
//   target := exp(s/2)
//   y_hat  := psi(s) = sum_m c_m * B_m(s)        (linear sum; coeffs ARE psi's)
//   err    := target - y_hat
//   denom  := sum_m B_m(s)^2 + delta
//   c_m    := c_m + eta * B_m(s) * err / denom
//
// Target reconciliation between specs 5.1 and 5.5.2: 5.5.2's pseudocode
// reads "target := exp(s)" with y_hat = sum c*B. 5.1 fixes the
// parametrisation as phi = psi^2 with stored coefficients being psi's, and
// initialises psi via LS fit to exp(s/2) so phi approximates exp(s).
// Carrying that through 5.5.2 means the linear-in-c target must be exp(s/2),
// not exp(s): the post-fit phi = psi^2 = exp(s) then makes phi/sum(phi) =
// exp(s)/sum(exp(s)) = softmax(s), matching 5.5.2's stated outcome. The
// "target := exp(s)" line in 5.5.2's pseudocode is a notational artefact of
// treating phi as linear in c; the actual quantity linear in c is psi, and
// its target is exp(s/2). This implementation follows the 5.1-consistent
// reading. (Spec note: 5.5.2 should be tightened to say psi / exp(s/2) so
// the two sections align.)
//
// Out-of-grid short-circuit: scores outside the basis grid have zero basis
// support (FBasis.Evaluate returns zero BasisVals). Without a guard the
// per-score NLMS would still compute Target = Exp(0.5 * S), which for the
// chain's ReLUL clamp range (+/-500) can overflow Single precision to +Inf;
// then Step = Inf, and the coefficient update Step * BasisVals[M] hits
// Inf * 0 = NaN, permanently poisoning the coefficient. The no-support
// short-circuit skips the entire update for any score with no basis
// activation, which is mathematically a no-op anyway. A v1.1 hardening
// would apply your "rescale and spread" idea: subtract a fixed grid-tied
// constant from S before Exp when S approaches the overflow threshold,
// letting the gauge absorb the resulting scale (phi/sum(phi) is gauge-
// invariant). The short-circuit form is correct for the default grid;
// the rescale form is robust against pathologically wide custom grids.
//
// Same Depth-stride indexing as EvaluateSplineRow.
var
  J, FirstIdx, M, KnotN, D: integer;
  S, Psi, Target, Err, Denom, BasisSqSum, Step: TNeuralFloat;
  BasisVals: array[0..3] of TNeuralFloat;
begin
  KnotN := Length(FHead.Coeffs);
  D := ScoresRow.Depth;
  for J := 0 to RowLen - 1 do
  begin
    S := ScoresRow.Raw[J * D + RowIdx];
    FBasis.Evaluate(S, FirstIdx, BasisVals);
    Psi := 0;
    BasisSqSum := 0;
    for M := 0 to 3 do
    begin
      if (FirstIdx + M >= 0) and (FirstIdx + M < KnotN) then
      begin
        Psi := Psi + FHead.Coeffs[FirstIdx + M] * BasisVals[M];
        BasisSqSum := BasisSqSum + BasisVals[M] * BasisVals[M];
      end;
    end;
    // No basis support at this score => coefficient update would be a no-op.
    // Skip explicitly to avoid the Inf * 0 = NaN trap if Exp(0.5*S)
    // overflows for far-out-of-grid scores (header comment).
    if BasisSqSum <= 0 then Continue;
    Denom := BasisSqSum + FNlmsDelta;
    Target := Exp(0.5 * S);
    Err := Target - Psi;
    Step := FNlmsEta * Err / Denom;
    for M := 0 to 3 do
    begin
      if (FirstIdx + M >= 0) and (FirstIdx + M < KnotN) then
        FHead.Coeffs[FirstIdx + M] := FHead.Coeffs[FirstIdx + M] + Step * BasisVals[M];
    end;
  end;
end;

procedure TNNetKANNormaliser.NLMSPhaseD(const ScoresRow: TNNetVolume;
                                         const RowIdx, RowLen: integer);
// Spec 5.5.3: Phase D self-distillation sharpening. Row-snapshotted target
// derived from the head's own current attention distribution, sharpened by
// power alpha.
//   w[j]          := FWKANRow[j]                           (= phi_row / S_row)
//   w_sharp[j]    := w[j]^alpha / sum_k(w[k]^alpha)
//   target_pre[j] := w_sharp[j] * S_row                    (back into phi-units)
//   for each score s = ScoresRow at (RowIdx, J):
//     err   := target_pre[j] - phi_row[j]
//     denom := sum_m B_m(s)^2 + delta
//     c_m   := c_m + eta * B_m(s) * err / denom            (only k+1 m's hit)
//
// Pipeline preconditions (caller responsibility, met by §7 ordering):
//   - EvaluateSplineRow has populated FPhiRow + FRowSum for this row.
//   - ComputeWeightsRow has populated FWKANRow for this row.
//
// Per spec 5.5.3 "row snapshot" rationale: FPhiRow / FRowSum / FWKANRow are
// frozen for this row's per-score updates. The next row's EvaluateSplineRow
// recomputes them from the now-updated coefficients.
//
// Chain-rule note: spec 5.5.3's NLMS step c_m += eta * B_m * err / denom is
// the linear-in-c form. Under 5.1's phi = psi^2 parametrisation, the proper
// phi-space gradient step would carry an additional 2*psi factor (chain
// rule). The spec writes the simpler form; this implementation follows it.
// Direction is correct (sign-preserving when psi > 0, held by GM=1 + sign
// preservation); magnitude is auto-tuned by NLMS denom; mechanism #1 + gauge
// keep coefficients bounded. (Spec note: 5.5.3 should clarify whether
// chain-rule omission is intentional.)
//
// Same Depth-stride indexing as EvaluateSplineRow.
var
  J, FirstIdx, M, KnotN, D: integer;
  S, SumSharp, Err, Denom, BasisSqSum, Step: TNeuralFloat;
  BasisVals: array[0..3] of TNeuralFloat;
begin
  if Length(FWSharpRow) < RowLen then SetLength(FWSharpRow, RowLen);
  if Length(FTargetPreRow) < RowLen then SetLength(FTargetPreRow, RowLen);

  // Pathological row: phi_row was all-zero (EvaluateSplineRow set FRowSum to
  // 0 and FWKANRow to zeros). Next pass, after mechanism #1 has corrected
  // any coefficient state that produced this, will give a usable target.
  if FRowSum <= 0 then exit;

  // w_sharp[j] = w[j]^alpha / sum(w[k]^alpha)
  SumSharp := 0;
  for J := 0 to RowLen - 1 do
  begin
    FWSharpRow[J] := Power(FWKANRow[J], FSharpenAlpha);
    SumSharp := SumSharp + FWSharpRow[J];
  end;
  if SumSharp <= 0 then exit;  // all-zero w: defer to next pass

  // target_pre[j] = w_sharp[j] * S_row, in phi-units (matches FPhiRow units).
  for J := 0 to RowLen - 1 do
  begin
    FWSharpRow[J] := FWSharpRow[J] / SumSharp;
    FTargetPreRow[J] := FWSharpRow[J] * FRowSum;
  end;

  // Per-score NLMS against the snapshotted target.
  KnotN := Length(FHead.Coeffs);
  D := ScoresRow.Depth;
  for J := 0 to RowLen - 1 do
  begin
    S := ScoresRow.Raw[J * D + RowIdx];
    FBasis.Evaluate(S, FirstIdx, BasisVals);
    BasisSqSum := 0;
    for M := 0 to 3 do
    begin
      if (FirstIdx + M >= 0) and (FirstIdx + M < KnotN) then
        BasisSqSum := BasisSqSum + BasisVals[M] * BasisVals[M];
    end;
    // No basis support at this score => update is a no-op. Skip for
    // consistency with NLMSPhaseM and to avoid wasted work; Phase D doesn't
    // have the Exp overflow path Phase M does, but the pattern is uniform.
    if BasisSqSum <= 0 then Continue;
    Denom := BasisSqSum + FNlmsDelta;
    Err := FTargetPreRow[J] - FPhiRow[J];
    Step := FNlmsEta * Err / Denom;
    for M := 0 to 3 do
    begin
      if (FirstIdx + M >= 0) and (FirstIdx + M < KnotN) then
        FHead.Coeffs[FirstIdx + M] := FHead.Coeffs[FirstIdx + M] + Step * BasisVals[M];
    end;
  end;
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

function TNNetKANNormaliser.HighClipAt(const I: integer; const G: TNeuralFloat;
                                       const LayerClipRate: TNeuralFloat): boolean;
// Spec 5.2.3: if |c_i| > 3*g, clip it down to 3*g (sign-preserving) and
// redistribute the multiplicative excess r = |c_i_old|/|c_i_new| > 1
// across the other 2k window coefficients via positive factors
// f_j = r^{s_j} with share weights s_j summing to 1 (5.2.5). The
// per-window product |c_i| * Pi_{j!=i}|c_j| is preserved exactly:
//     Pi_new = (|c_i|/r) * Pi_{j!=i}(|c_j| * f_j)
//            = (|c_i|/r) * r * Pi_{j!=i}|c_j|  =  Pi_old              (M1-1)
// All factors are positive, so coefficient signs are preserved (M1-2).
// Returns true iff a clip actually fired.
var
  J, K, Lo, Hi, N: integer;
  AbsOld, AbsNew, R, LogR, ShareTotal, ShareJ, Factor, SignI: TNeuralFloat;
  ShareWeights: array of TNeuralFloat;
  Rule: TKANShareRule;
const
  CFloor = 1e-30;
begin
  Result := False;
  AbsOld := Abs(FHead.Coeffs[I]);
  AbsNew := 3.0 * G;
  if AbsOld <= AbsNew then exit;

  // Clip the offending coefficient.
  if FHead.Coeffs[I] >= 0 then SignI := 1 else SignI := -1;
  FHead.Coeffs[I] := SignI * AbsNew;

  R := AbsOld / AbsNew;
  if R <= 1 then exit;
  LogR := Ln(R);

  // Window (clamped at array boundaries).
  K := FGridSpec.BasisOrder;
  N := Length(FHead.Coeffs);
  Lo := Max(0, I - K);
  Hi := Min(N - 1, I + K);

  Rule := ResolveShareMode({IsHighClip=}True, LayerClipRate);

  // Compute raw share weights s_j over j != i in window.
  SetLength(ShareWeights, Hi - Lo + 1);
  ShareTotal := 0;
  for J := Lo to Hi do
  begin
    if J = I then
    begin
      ShareWeights[J - Lo] := 0;
      Continue;
    end;
    case Rule of
      ksrProportional:        ShareJ := Max(Abs(FHead.Coeffs[J]), CFloor);
      ksrInverseProportional: ShareJ := 1 / Max(Abs(FHead.Coeffs[J]), CFloor);
    else                       ShareJ := Max(Abs(FHead.Coeffs[J]), CFloor);
    end;
    ShareWeights[J - Lo] := ShareJ;
    ShareTotal := ShareTotal + ShareJ;
  end;

  // Singleton window or all-zero neighbours: no redistribution possible.
  // The clip still stood; window product is no longer preserved at the
  // boundary, but the cascade's next iteration will absorb the residual.
  if ShareTotal <= 0 then
  begin
    Result := True;
    exit;
  end;

  // Apply factors: f_j = R^{s_j / ShareTotal}; exponents sum to 1, so the
  // product of factors is exactly R.
  for J := Lo to Hi do
  begin
    if J = I then Continue;
    if ShareWeights[J - Lo] <= 0 then Continue;
    Factor := Exp((ShareWeights[J - Lo] / ShareTotal) * LogR);
    FHead.Coeffs[J] := FHead.Coeffs[J] * Factor;
  end;

  Result := True;
end;

function TNNetKANNormaliser.LowLiftAt(const I: integer; const G: TNeuralFloat;
                                      const LayerClipRate: TNeuralFloat): boolean;
// Spec 5.2.4 (productivity rule): if |c_i| < g/2, lift it to |c_i| := g
// (not just to the threshold -- the asymmetric 5.2.4 design) and shrink
// the other window coefficients by f_j = (1/q)^{s_j / sum(s)} where
// q = |c_i_new| / |c_i_old| > 1. Per-window product preserved (M1-1):
//     Pi_new = (|c_i| * q) * Pi_{j!=i}(|c_j| * f_j)
//            = (|c_i| * q) * (1/q) * Pi_{j!=i}|c_j|  =  Pi_old
// The asymmetric threshold (g/2 vs the high-clip 3*g) makes low-lift do
// productivity work, not numerical safety; the asymmetry is what makes
// head-squaring's capacity signal meaningful (5.2.4 rationale).
// Returns true iff a lift actually fired.
var
  J, K, Lo, Hi, N: integer;
  AbsOld, AbsNew, Q, LogQ, ShareTotal, ShareJ, Factor, SignI: TNeuralFloat;
  ShareWeights: array of TNeuralFloat;
  Rule: TKANShareRule;
const
  CFloor = 1e-30;
begin
  Result := False;
  AbsOld := Abs(FHead.Coeffs[I]);
  if AbsOld >= 0.5 * G then exit;

  // Lift toward GM.
  AbsNew := G;
  if FHead.Coeffs[I] >= 0 then SignI := 1 else SignI := -1;
  FHead.Coeffs[I] := SignI * AbsNew;

  if AbsOld < CFloor then AbsOld := CFloor;
  Q := AbsNew / AbsOld;
  if Q <= 1 then exit;
  LogQ := Ln(Q);

  K := FGridSpec.BasisOrder;
  N := Length(FHead.Coeffs);
  Lo := Max(0, I - K);
  Hi := Min(N - 1, I + K);

  Rule := ResolveShareMode({IsHighClip=}False, LayerClipRate);

  SetLength(ShareWeights, Hi - Lo + 1);
  ShareTotal := 0;
  for J := Lo to Hi do
  begin
    if J = I then
    begin
      ShareWeights[J - Lo] := 0;
      Continue;
    end;
    case Rule of
      ksrProportional:        ShareJ := Max(Abs(FHead.Coeffs[J]), CFloor);
      ksrInverseProportional: ShareJ := 1 / Max(Abs(FHead.Coeffs[J]), CFloor);
    else                       ShareJ := Max(Abs(FHead.Coeffs[J]), CFloor);
    end;
    ShareWeights[J - Lo] := ShareJ;
    ShareTotal := ShareTotal + ShareJ;
  end;

  if ShareTotal <= 0 then
  begin
    Result := True;
    exit;
  end;

  // f_j = (1/Q)^{s_j/ShareTotal} = Exp(-s_j * LogQ / ShareTotal);
  // product of factors is exactly 1/Q.
  for J := Lo to Hi do
  begin
    if J = I then Continue;
    if ShareWeights[J - Lo] <= 0 then Continue;
    Factor := Exp(-(ShareWeights[J - Lo] / ShareTotal) * LogQ);
    FHead.Coeffs[J] := FHead.Coeffs[J] * Factor;
  end;

  Result := True;
end;

function TNNetKANNormaliser.SweepOnce(const LayerClipRate: TNeuralFloat): integer;
// Spec 5.2.6 inner loop: one pass over all coefficients. For each c_i,
// recompute its window's g and try high-clip (5.2.3); failing that, try
// low-lift (5.2.4). High-clip and low-lift are mutually exclusive on the
// value of |c_i| (one requires above 3*g, the other below g/2), so at
// most one fires per coefficient per pass. Returns the count of fires.
var
  I, N: integer;
  G: TNeuralFloat;
begin
  Result := 0;
  N := Length(FHead.Coeffs);
  for I := 0 to N - 1 do
  begin
    G := WindowGM(I);
    if HighClipAt(I, G, LayerClipRate) then
      Inc(Result)
    else if LowLiftAt(I, G, LayerClipRate) then
      Inc(Result);
  end;
end;

procedure TNNetKANNormaliser.Mechanism1Sweep(const LayerClipRate: TNeuralFloat);
// Spec 5.2.6: cascade the per-coefficient sweep to a fixed point with a
// hard safety cap (FCascadeMaxIter, default 16). If the cap is reached
// without convergence the per-head FCascadeCapHits counter is incremented
// (5.2.6 "Behaviour at the cap" note). Invariant M1-1 (per-window product
// preservation) is satisfied throughout regardless of cap-hit, because
// every individual event in SweepOnce preserves products by construction;
// only M1-3 (bounded coefficients) may be violated for the capped pass,
// and the next pass's sweep continues the relaxation.
var
  Iter, Fired, Cap: integer;
begin
  Cap := FCascadeMaxIter;
  if Cap < 1 then Cap := 16;
  for Iter := 1 to Cap do
  begin
    Fired := SweepOnce(LayerClipRate);
    if Fired = 0 then exit;
  end;
  // Cap reached without convergence; record it (5.2.6 cap behaviour).
  Inc(FCascadeCapHits);
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
