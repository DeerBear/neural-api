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

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)

unit neuralkanattention;

(*
KAN Attention Normaliser — drop-in replacement for the softmax operator
inside multi-head self-attention. Trained as standard softmax; at inference
each per-head B-spline normaliser learns to mimic softmax then takes over
per-layer per-head and evolves to surface information softmax would lose.

Mechanism specification:    docs/kan_attention_spec.md
Implementation decisions:   docs/kan_implementation_pascal.md

v1 status: SKELETON. Method bodies are stubs that raise EKANBadState.
*)

{$include neuralnetwork.inc}

interface

uses
  {$IFDEF FPC}
  fgl,
  {$ENDIF}
  Classes, SysUtils, Math,
  neuralvolume, neuralnetwork;

type
  // ===================================================================
  //  EXCEPTIONS
  // ===================================================================

  /// Raised when a training-only operation is attempted while the network
  /// is locked to inference (see TKANNet.LockToInference).
  EKANInInference = class(Exception);

  /// Raised when an inference-only operation is attempted while the
  /// network is still in training mode.
  EKANNotLocked = class(Exception);

  /// Raised on impossible internal states (defensive).
  EKANBadState = class(Exception);

  // ===================================================================
  //  CONFIGURATION ENUMS  (spec §4, §5.2.5, §11.8)
  // ===================================================================

  /// Per-head lifecycle phase. One-way transition Softmax -> KAN.
  TKANStatus = (ksSoftmaxActive, ksKANActive);

  /// Mechanism #1 share rules (spec §5.2.5).
  TKANShareRule = (ksrProportional, ksrInverseProportional, ksrAuto);

  /// Mode override for ablation / debugging (spec §11.8).
  /// Mode D is structurally excluded by the Auto logic and intentionally absent.
  TKANForceMode = (kfmAuto, kfmA, kfmB, kfmC);

  // ===================================================================
  //  GRID SPECIFICATION  (spec §5.1)
  // ===================================================================

  /// Immutable B-spline grid configuration. Set at layer construction;
  /// never modified for the lifetime of a layer.
  TKANGridSpec = record
    GridLow, GridHigh: TNeuralFloat;   // grid range in nats; default [-8, +8]
    KnotCount: integer;                // N; default 64
    BasisOrder: integer;               // k; default 3 (cubic)
    function Hash: UInt64;             // for deterministic per-layer RNG seeding
  end;

  // ===================================================================
  //  PER-HEAD STATE  (spec §5.1, §6, §8.1)
  // ===================================================================

  /// All mutable per-head spline state. One instance lives inside each
  /// TNNetKANNormaliser.
  TKANHeadState = record
    Coeffs: array of TNeuralFloat;     // psi-space; length = TKANGridSpec.KnotCount
    Status: TKANStatus;
    KLEMA: TNeuralFloat;
    ConsecutiveLowPasses: integer;
    ClipCountTotal: integer;
  end;

  // ===================================================================
  //  SEEDED RNG  (spec §9)
  // ===================================================================

  /// SplitMix64 deterministic RNG. Required because neural-api otherwise
  /// uses global Random() which does not satisfy spec §9 determinism.
  TKANSeededRNG = record
    State: UInt64;
    procedure Seed(const s: UInt64);
    function NextU64: UInt64;
    function NextFloat: TNeuralFloat;       // uniform [0, 1)
    function NextNormal: TNeuralFloat;      // standard normal via Box-Muller
  end;
  PKANSeededRNG = ^TKANSeededRNG;

  // ===================================================================
  //  B-SPLINE BASIS  (spec §5.1)
  // ===================================================================

  /// Immutable B-spline basis. One instance per attention layer, shared
  /// across all H_max heads (the grid is identical for all heads).
  TKANBasis = class
  private
    FGridSpec: TKANGridSpec;
    FKnots: array of TNeuralFloat;
  public
    constructor Create(const Spec: TKANGridSpec);
    destructor Destroy; override;

    /// Evaluates the k+1 non-zero basis values at score s.
    /// Writes to Vals[0..k]; returns the index of the first active basis.
    procedure Evaluate(const s: TNeuralFloat;
                       out FirstIdx: integer;
                       Vals: PSingle);

    /// NLMS denominator term: Σ_j B_j(s)² over the k+1 active basis functions.
    function BasisSquaredSum(const s: TNeuralFloat): TNeuralFloat;

    /// Cold-start fit: returns psi-coefficients such that ψ(s)² ≈ exp(s)
    /// over the grid range. Least-squares on (s, exp(s/2)) pairs (spec §5.1, §8.4).
    procedure FitPsiToExp(out Coeffs: array of TNeuralFloat);

    property GridSpec: TKANGridSpec read FGridSpec;
  end;

  // ===================================================================
  //  LAYER CLASS  (spec §5, §6, §7)
  // ===================================================================

  TKANNet = class;  // forward declaration

  /// Per-head B-spline normaliser. Drop-in replacement for the
  /// per-head TNNetPointwiseSoftMax inside multi-head self-attention.
  /// Inherits TNNetIdentity for output-shape and gradient-routing plumbing.
  TNNetKANNormaliser = class(TNNetIdentity)
  private
    // --- Identity ---
    FAttentionLayerId: integer;
    FHeadIndex: integer;

    // --- Shared resources (owned by TKANNet's per-attention-layer metadata) ---
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

    // --- Per-pass scratch (allocated in constructor) ---
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

  // ===================================================================
  //  PER-ATTENTION-LAYER METADATA
  // ===================================================================

  /// Tracks the H_max normalisers and shared resources for one
  /// attention layer in a TKANNet. Owned by the network.
  TKANAttentionLayerInfo = class
  private
    FAttentionLayerId: integer;
    FBasis: TKANBasis;
    FRNG: TKANSeededRNG;
    FActiveHeads: integer;
    FClipRateEMA: TNeuralFloat;
    FConsecutiveHighSweeps: integer;
    FNormalisers: TList;               // of TNNetKANNormaliser; length = H_max
  public
    constructor Create(const AttentionLayerId: integer;
                       const Spec: TKANGridSpec;
                       const HMax: integer;
                       const Seed: UInt64);
    destructor Destroy; override;

    procedure RegisterNormaliser(const N: TNNetKANNormaliser);

    /// Per-attention-layer post-batch hook: reads each normaliser's
    /// per-head clip activity, updates FClipRateEMA, fires squaring if
    /// criterion is met (spec §5.4.3, §7 step 11).
    procedure CheckSquaring;

    property AttentionLayerId: integer read FAttentionLayerId;
    property Basis: TKANBasis read FBasis;
    property ActiveHeads: integer read FActiveHeads;
    property ClipRateEMA: TNeuralFloat read FClipRateEMA;
  end;

  // ===================================================================
  //  NETWORK CLASS  (spec §3, §11, §14; impl doc §4, §6)
  // ===================================================================

  /// Network subclass that owns the KAN attention infrastructure:
  /// registry of KAN layers, mode lock, bulk operations, and the
  /// AddKANSelfAttention builder.
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
    function  KANEnabledMask: TBooleanDynArray;

    // --- Telemetry ---
    function  KANTelemetry: string;

    // --- Training-method overrides that gate on lock state ---
    procedure Backpropagate(pInput: TNNetVolume); override;

    property InferenceLocked: boolean read FInferenceLocked;
  end;

implementation

// =====================================================================
//  TKANGridSpec
// =====================================================================

function TKANGridSpec.Hash: UInt64;
begin
  // SplitMix64-style mix of all four fields. Stable across runs.
  Result := UInt64(KnotCount) * UInt64($9E3779B97F4A7C15);
  Result := Result xor (UInt64(BasisOrder) * UInt64($BF58476D1CE4E5B9));
  Result := Result xor (UInt64(Trunc(GridLow * 1000)) * UInt64($94D049BB133111EB));
  Result := Result xor (UInt64(Trunc(GridHigh * 1000)) * UInt64($D6E8FEB86659FD93));
end;

// =====================================================================
//  TKANSeededRNG  (SplitMix64)
// =====================================================================

procedure TKANSeededRNG.Seed(const s: UInt64);
begin
  State := s;
end;

function TKANSeededRNG.NextU64: UInt64;
begin
  State := State + UInt64($9E3779B97F4A7C15);
  Result := State;
  Result := (Result xor (Result shr 30)) * UInt64($BF58476D1CE4E5B9);
  Result := (Result xor (Result shr 27)) * UInt64($94D049BB133111EB);
  Result := Result xor (Result shr 31);
end;

function TKANSeededRNG.NextFloat: TNeuralFloat;
begin
  // 53-bit-equivalent uniform [0, 1); cast down to Single for TNeuralFloat.
  Result := TNeuralFloat((NextU64 shr 11) / TNeuralFloat($1FFFFFFFFFFFFF));
end;

function TKANSeededRNG.NextNormal: TNeuralFloat;
var
  u1, u2: TNeuralFloat;
begin
  // Box-Muller. Inefficient (discards the second sample); acceptable since
  // the only caller is head-squaring perturbation, called rarely.
  u1 := NextFloat;
  u2 := NextFloat;
  if u1 < 1e-30 then u1 := 1e-30;
  Result := Sqrt(-2.0 * Ln(u1)) * Cos(2.0 * Pi * u2);
end;

// =====================================================================
//  TKANBasis
// =====================================================================

constructor TKANBasis.Create(const Spec: TKANGridSpec);
begin
  inherited Create;
  FGridSpec := Spec;
  // TODO: precompute knot positions (uniform on [GridLow, GridHigh]).
  raise EKANBadState.Create('TKANBasis.Create: not implemented');
end;

destructor TKANBasis.Destroy;
begin
  SetLength(FKnots, 0);
  inherited Destroy;
end;

procedure TKANBasis.Evaluate(const s: TNeuralFloat;
                              out FirstIdx: integer;
                              Vals: PSingle);
begin
  // TODO: cubic B-spline evaluation, k+1 active basis values.
  raise EKANBadState.Create('TKANBasis.Evaluate: not implemented');
end;

function TKANBasis.BasisSquaredSum(const s: TNeuralFloat): TNeuralFloat;
begin
  // TODO: Σ_j B_j(s)² over the k+1 active basis functions.
  raise EKANBadState.Create('TKANBasis.BasisSquaredSum: not implemented');
end;

procedure TKANBasis.FitPsiToExp(out Coeffs: array of TNeuralFloat);
begin
  // TODO: least-squares fit ψ(s) ≈ exp(s/2) on a dense grid;
  // resulting coefficients give φ = ψ² ≈ exp(s) over [GridLow, GridHigh].
  raise EKANBadState.Create('TKANBasis.FitPsiToExp: not implemented');
end;

// =====================================================================
//  TNNetKANNormaliser
// =====================================================================

constructor TNNetKANNormaliser.Create(const GridSpec: TKANGridSpec;
                                       const AttentionLayerId, HeadIndex: integer;
                                       const SharedBasis: TKANBasis;
                                       const SharedRNG: PKANSeededRNG);
begin
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
  inherited Backpropagate;
end;

procedure TNNetKANNormaliser.ComputeAsSoftmax;
begin
  // TODO: reproduce TNNetPointwiseSoftMax.Compute behaviour on FPrevLayer.Output
  // Standard row-wise softmax to FOutput.
  raise EKANBadState.Create('TNNetKANNormaliser.ComputeAsSoftmax: not implemented');
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
  // (spec §5.3)
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
  // TODO: write zeros to FOutput so the downstream concat receives
  // zero contribution from this head (HS-3).
  FOutput.Fill(0);
end;

// =====================================================================
//  TKANAttentionLayerInfo
// =====================================================================

constructor TKANAttentionLayerInfo.Create(const AttentionLayerId: integer;
                                           const Spec: TKANGridSpec;
                                           const HMax: integer;
                                           const Seed: UInt64);
begin
  inherited Create;
  FAttentionLayerId := AttentionLayerId;
  FBasis := TKANBasis.Create(Spec);
  FRNG.Seed(Seed);
  FActiveHeads := 2;                    // spec-mandated initial value (§5.4.2)
  FClipRateEMA := 0;
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

procedure TKANAttentionLayerInfo.CheckSquaring;
begin
  // TODO: aggregate per-head clip activity into FClipRateEMA;
  // if FClipRateEMA > τ_squaring for K_squaring consecutive sweeps,
  // fire DoSquaring (spec §5.4.4).
  raise EKANBadState.Create('TKANAttentionLayerInfo.CheckSquaring: not implemented');
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
begin
  AssertNotLocked('AddKANSelfAttention');

  // TODO:
  //   1. Compute H_max := largest_pow2 <= ⌊d/2⌋ if HeadCeiling = 0
  //   2. Build TKANGridSpec; create TKANAttentionLayerInfo with seeded RNG
  //   3. Mirror AddSelfAttention's chain construction:
  //        Q, K, V projections -> scale -> split into H_max heads
  //   4. For each of H_max heads: insert TNNetKANNormaliser (instead of softmax)
  //   5. Concat H_max heads (DeepConcat); inactive heads contribute zero via mask
  //   6. Apply output projection W_O
  //   7. Register normalisers in TKANAttentionLayerInfo
  //   8. Return final layer

  Result := nil;
  raise EKANBadState.Create('TKANNet.AddKANSelfAttention: not implemented');
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
    for j := 0 to Info.FNormalisers.Count - 1 do
    begin
      Normaliser := TNNetKANNormaliser(Info.FNormalisers[j]);
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
  for j := 0 to Info.FNormalisers.Count - 1 do
    TNNetKANNormaliser(Info.FNormalisers[j]).KANEnabled := false;
end;

procedure TKANNet.EnableKANForLayer(const Idx: integer);
var
  Info: TKANAttentionLayerInfo;
  j: integer;
begin
  if (Idx < 0) or (Idx >= FAttentionLayers.Count) then
    raise EKANBadState.CreateFmt('EnableKANForLayer: index %d out of range', [Idx]);
  Info := TKANAttentionLayerInfo(FAttentionLayers[Idx]);
  for j := 0 to Info.FNormalisers.Count - 1 do
    TNNetKANNormaliser(Info.FNormalisers[j]).KANEnabled := true;
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

function TKANNet.KANEnabledMask: TBooleanDynArray;
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
    Result[i] := (Info.FNormalisers.Count > 0)
                 and TNNetKANNormaliser(Info.FNormalisers[0]).KANEnabled;
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
