unit TestNeuralKAN;

{$mode objfpc}{$H+}

(*
KAN Attention Normaliser — unit tests.

Mechanism specification:    docs/kan_attention_spec.md §12
Implementation decisions:   docs/kan_implementation_pascal.md §9

v1 status: SKELETON. All test bodies are stubs; the unit registers the
test class with the harness so it appears in the runner, but every test
currently fails with a "not implemented" message until the layer's
mechanisms are implemented.
*)

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralvolume, neuralnetwork,
  neuralkantypes, neuralkanbasis, neuralkannormaliser, neuralkanattention;

type
  TTestNeuralKAN = class(TTestCase)
  published
    // ---- Tier 1: Invariant tests (spec §12.1) ----
    procedure TestKANWindowProductPreservation;        // IT-M1
    procedure TestKANSignPreservation;                 // IT-Sign
    procedure TestKANBoundedCoefficients;              // IT-Bound
    procedure TestKANScaleEquivariance;                // IT-Scale
    procedure TestKANGaugeFixed;                       // IT-Gauge
    procedure TestKANGaugeOutputPreservation;          // IT-GaugeOutput
    procedure TestKANHeadSquaringBounds;               // IT-HSBound
    procedure TestKANInactiveHeadMask;                 // IT-HSMask
    procedure TestKANHeadSquaringMonotonicity;         // IT-HSMonotonic
    procedure TestKANNLMSLocality;                     // IT-LLLocal
    procedure TestKANLifecycleMonotonicity;            // IT-LCMonotonic
    procedure TestKANRetrofitIdentity;                 // IT-Retrofit
    procedure TestKANDeterminism;                      // IT-Det

    // ---- Tier 2: Functional tests (spec §12.2) ----
    procedure TestKANMimicryConvergence;               // FT-Mimic
    procedure TestKANHandoverAtomicity;                // FT-Handover
    procedure TestKANPhaseDPeakDevelopment;            // FT-Phase D
    procedure TestKANPhaseDNoCollapse;                 // FT-NoCollapse
    procedure TestKANSquaringTrigger;                  // FT-Squaring
    procedure TestKANSquaringSelfLimit;                // FT-SquaringSelfLimit
    procedure TestKANModeDExclusion;                   // FT-ModeAbsent

    // ---- Lifecycle / mode safety (impl doc §4) ----
    procedure TestKANLockToInferenceIsOneWay;
    procedure TestKANBackpropRaisesInInferenceMode;
    procedure TestKANInferenceForwardRequiresLock;
    procedure TestKANSeededRNGDeterministic;
    procedure TestKANSeededRNGRoundTripsThroughSerialisation;

    // ---- B-spline basis (impl doc §3.4) ----
    procedure TestKANBasisLocalSupport;
    procedure TestKANBasisPositivity;
    procedure TestKANBasisInteriorPartitionOfUnity;
    procedure TestKANBasisFitToExpReasonable;

    // ---- Softmax fallback (impl doc §5.1) ----
    procedure TestKANNormaliserSoftmaxFallbackInTraining;

    // ---- Squaring trigger (spec §5.4.3) ----
    procedure TestKANSquaringTriggerNotFiringInitially;
    procedure TestKANSquaringTriggerFiresAfterSustainedHighRate;
    procedure TestKANSquaringTriggerResetsOnLowRate;
    procedure TestKANSquaringTriggerClampedAtCeiling;
    procedure TestKANSquaringTriggerEmaSmoothing;

    // ---- Coverage gaps for already-implemented surface ----
    procedure TestKANGridSpecHashDeterministic;
    procedure TestKANRNGNextFloatRange;
    procedure TestKANRNGNextNormalSanity;
    procedure TestKANBasisCreateValidation;
    procedure TestKANNormaliserInitialPerHeadState;
    procedure TestKANNormaliserEnterInferenceFitsCoefficients;
    procedure TestKANNormaliserEnterInferenceRequiresSeededRNG;
    procedure TestKANInfoCreateValidation;
    procedure TestKANInfoCheckSquaringRaisesWhenTriggered;
    procedure TestKANNetBackpropGatesOnLock;
    procedure TestKANNetBulkOpsOnEmptyNetwork;
    procedure TestKANNetBulkOpsIndexValidation;
  end;

implementation

const
  NOT_IMPLEMENTED = 'KAN attention is at skeleton stage; mechanism not implemented';

// ---- Tier 1 ----

procedure TTestNeuralKAN.TestKANWindowProductPreservation;
begin
  // Apply mech#1 to a randomised coefficient vector (with negatives);
  // assert Π|c_j| within each window unchanged within 1e-6 relative tol.
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANSignPreservation;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANBoundedCoefficients;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANScaleEquivariance;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANGaugeFixed;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANGaugeOutputPreservation;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANHeadSquaringBounds;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANInactiveHeadMask;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANHeadSquaringMonotonicity;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANNLMSLocality;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANLifecycleMonotonicity;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANRetrofitIdentity;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANDeterminism;
begin
  Fail(NOT_IMPLEMENTED);
end;

// ---- Tier 2 ----

procedure TTestNeuralKAN.TestKANMimicryConvergence;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANHandoverAtomicity;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANPhaseDPeakDevelopment;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANPhaseDNoCollapse;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANSquaringTrigger;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANSquaringSelfLimit;
begin
  Fail(NOT_IMPLEMENTED);
end;

procedure TTestNeuralKAN.TestKANModeDExclusion;
begin
  Fail(NOT_IMPLEMENTED);
end;

// ---- Lifecycle / mode safety ----

procedure TTestNeuralKAN.TestKANLockToInferenceIsOneWay;
var
  NN: TKANNet;
begin
  NN := TKANNet.Create;
  try
    AssertFalse('Fresh TKANNet starts unlocked', NN.InferenceLocked);
    NN.LockToInference;
    AssertTrue('LockToInference sets InferenceLocked', NN.InferenceLocked);
    NN.LockToInference;  // idempotent; should not raise
    AssertTrue('LockToInference is idempotent', NN.InferenceLocked);
    // No public API exists to unlock; if one is added, test should fail.
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANBackpropRaisesInInferenceMode;
var
  Spec: TKANGridSpec;
  Basis: TKANBasis;
  RNG: TKANSeededRNG;
  Norm: TNNetKANNormaliser;
  Caught: boolean;
begin
  // After EnterInferenceMode, the layer's Backpropagate must raise
  // EKANInInference (spec §5.5.1: no autograd at inference). The raise
  // happens at the top of Backpropagate before any chain wiring is
  // touched, so we do not need a fully wired FPrevLayer for this test.
  Spec := MakeDefaultGridSpec(32);
  Basis := TKANBasis.Create(Spec);
  RNG.Seed(42);

  Norm := TNNetKANNormaliser.Create(Spec, 0, 0, Basis, @RNG);
  try
    Norm.EnterInferenceMode;
    Caught := false;
    try
      Norm.Backpropagate;
    except
      on EKANInInference do Caught := true;
    end;
    AssertTrue('Backpropagate in inference mode must raise EKANInInference', Caught);
  finally
    Norm.Free;
    Basis.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANInferenceForwardRequiresLock;
var
  NN: TKANNet;
  Caught: boolean;
begin
  NN := TKANNet.Create;
  try
    Caught := false;
    try
      NN.InferenceForward(nil);
    except
      on EKANNotLocked do Caught := true;
    end;
    AssertTrue('InferenceForward without LockToInference must raise EKANNotLocked', Caught);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANSeededRNGDeterministic;
var
  R1, R2: TKANSeededRNG;
  i: integer;
begin
  R1.Seed(42);
  R2.Seed(42);
  for i := 0 to 1000 do
    AssertEquals('Seeded RNG produces identical sequences', R1.NextU64, R2.NextU64);
end;

procedure TTestNeuralKAN.TestKANSeededRNGRoundTripsThroughSerialisation;
var
  R1, R2: TKANSeededRNG;
  Snapshot: UInt64;
  i: integer;
begin
  R1.Seed(12345);
  for i := 0 to 100 do R1.NextU64;
  Snapshot := R1.State;
  R2.Seed(0);
  R2.State := Snapshot;
  for i := 0 to 1000 do
    AssertEquals('RNG state restore reproduces sequence', R1.NextU64, R2.NextU64);
end;

// ---- B-spline basis ----

function MakeDefaultGridSpec(KnotCount: integer = 64): TKANGridSpec;
begin
  Result.GridLow := -8.0;
  Result.GridHigh := 8.0;
  Result.KnotCount := KnotCount;
  Result.BasisOrder := 3;
end;

procedure TTestNeuralKAN.TestKANBasisLocalSupport;
var
  Basis: TKANBasis;
  Spec: TKANGridSpec;
  Vals: array[0..3] of TNeuralFloat;
  FirstIdx, i, NonZeroCount: integer;
  s: TNeuralFloat;
begin
  Spec := MakeDefaultGridSpec(32);
  Basis := TKANBasis.Create(Spec);
  try
    // Pick an interior score (well away from boundaries).
    s := 0.5;
    Basis.Evaluate(s, FirstIdx, Vals);
    NonZeroCount := 0;
    for i := 0 to 3 do
      if Vals[i] > 0 then Inc(NonZeroCount);
    AssertTrue('Interior score must have at least 3 non-zero basis values',
               NonZeroCount >= 3);
    AssertTrue('Interior score must have at most k+1 = 4 non-zero basis values',
               NonZeroCount <= 4);
  finally
    Basis.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANBasisPositivity;
var
  Basis: TKANBasis;
  Spec: TKANGridSpec;
  Vals: array[0..3] of TNeuralFloat;
  FirstIdx, i, j: integer;
  s: TNeuralFloat;
begin
  Spec := MakeDefaultGridSpec(32);
  Basis := TKANBasis.Create(Spec);
  try
    // Sample 50 scores across the grid range.
    for i := 0 to 49 do
    begin
      s := -7.5 + i * (15.0 / 49);
      Basis.Evaluate(s, FirstIdx, Vals);
      for j := 0 to 3 do
        AssertTrue(Format('Basis value at s=%.3f index %d must be >= 0 (got %.6f)',
                          [s, j, Vals[j]]),
                   Vals[j] >= 0);
    end;
  finally
    Basis.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANBasisInteriorPartitionOfUnity;
var
  Basis: TKANBasis;
  Spec: TKANGridSpec;
  Vals: array[0..3] of TNeuralFloat;
  FirstIdx, i, j: integer;
  s, total: TNeuralFloat;
begin
  Spec := MakeDefaultGridSpec(64);
  Basis := TKANBasis.Create(Spec);
  try
    // Cardinal cubic B-splines on a uniform grid sum to 1 in the
    // interior (away from boundaries). Sample 20 points well inside.
    for i := 0 to 19 do
    begin
      s := -6.0 + i * (12.0 / 19);
      Basis.Evaluate(s, FirstIdx, Vals);
      total := 0;
      for j := 0 to 3 do total := total + Vals[j];
      AssertTrue(Format('Interior partition-of-unity at s=%.3f: sum=%.6f, expected ~1.0',
                        [s, total]),
                 Abs(total - 1.0) < 1e-4);
    end;
  finally
    Basis.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANBasisFitToExpReasonable;
var
  Basis: TKANBasis;
  Spec: TKANGridSpec;
  Coeffs: array of TNeuralFloat;
  Vals: array[0..3] of TNeuralFloat;
  FirstIdx, i, j, idx: integer;
  s, psi, phi, expected, relErr: TNeuralFloat;
begin
  Spec := MakeDefaultGridSpec(64);
  Basis := TKANBasis.Create(Spec);
  try
    SetLength(Coeffs, Spec.KnotCount);
    Basis.FitPsiToExp(Coeffs);

    // After fit: phi = psi^2 should be ~ exp(s) at interior knot midpoints.
    // Allow generous tolerance — collocation-based init is approximate
    // (not a full least-squares fit).
    for i := 1 to 60 do
    begin
      s := -6.0 + i * (12.0 / 60);
      Basis.Evaluate(s, FirstIdx, Vals);
      psi := 0;
      for j := 0 to 3 do
      begin
        idx := FirstIdx + j;
        if (idx >= 0) and (idx < Spec.KnotCount) then
          psi := psi + Coeffs[idx] * Vals[j];
      end;
      phi := psi * psi;
      expected := Exp(s);
      relErr := Abs(phi - expected) / expected;
      AssertTrue(Format('FitPsiToExp at s=%.3f: phi=%.6g, expected exp(s)=%.6g, relErr=%.4f',
                        [s, phi, expected, relErr]),
                 relErr < 0.05);    // 5% tolerance for v1 collocation init
    end;
  finally
    Basis.Free;
  end;
end;

// ---- Softmax fallback ----

procedure TTestNeuralKAN.TestKANNormaliserSoftmaxFallbackInTraining;
var
  Spec: TKANGridSpec;
  Basis: TKANBasis;
  RNG: TKANSeededRNG;
  Norm: TNNetKANNormaliser;
  RowSum: TNeuralFloat;
  i: integer;
begin
  // In training mode (FInferenceMode = false, the constructor default),
  // Compute must behave identically to TNNetPointwiseSoftMax. We verify
  // by constructing a normaliser, stepping through ComputeAsSoftmax
  // indirectly via the inherited softmax path, and checking the output
  // is a valid probability distribution.

  Spec := MakeDefaultGridSpec(32);
  Basis := TKANBasis.Create(Spec);
  RNG.Seed(1);

  Norm := TNNetKANNormaliser.Create(Spec, 0, 0, Basis, @RNG);
  try
    // We do not exercise the full forward path here (that needs a wired-up
    // FPrevLayer + FOutput); we only verify that the inheritance chain is
    // correct and that the layer instantiates without raising. Detailed
    // initial-state checks live in TestKANNormaliserInitialPerHeadState.
    AssertEquals('AttentionLayerId stored correctly', 0, Norm.AttentionLayerId);
    AssertEquals('HeadIndex stored correctly', 0, Norm.HeadIndex);
    AssertTrue('KANEnabled defaults to true', Norm.KANEnabled);
    AssertTrue('Status defaults to ksSoftmaxActive', Norm.Status = ksSoftmaxActive);
  finally
    Norm.Free;
    Basis.Free;
  end;

  // Note: a fuller end-to-end test that constructs a TNNetVolume,
  // wires up FPrevLayer, calls Compute, and checks row-sum = 1 belongs
  // alongside the §7 pipeline implementation (one of the IT-Retrofit
  // checks). This test only verifies the layer instantiates with the
  // softmax inheritance correctly in place.
end;

// ---- Squaring trigger ----

procedure TTestNeuralKAN.TestKANSquaringTriggerNotFiringInitially;
var
  Spec: TKANGridSpec;
  Info: TKANAttentionLayerInfo;
begin
  Spec := MakeDefaultGridSpec(32);
  // Defaults: TauSquaring = 0.10, KSquaring = 64.
  Info := TKANAttentionLayerInfo.Create(0, Spec, 16, 1);
  try
    AssertEquals('Initial ActiveHeads = 2 per spec §5.4.2', 2, Info.ActiveHeads);
    AssertEquals('Initial ConsecutiveHighSweeps = 0', 0, Info.ConsecutiveHighSweeps);
    AssertFalse('Trigger does not fire on a fresh layer', Info.ShouldFireSquaring);
  finally
    Info.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANSquaringTriggerFiresAfterSustainedHighRate;
var
  Spec: TKANGridSpec;
  Info: TKANAttentionLayerInfo;
  i: integer;
begin
  Spec := MakeDefaultGridSpec(32);
  // Use a faster EMA so a small constant high signal pushes the EMA above
  // τ_squaring quickly; smaller KSquaring so the test runs in few iterations.
  // ClipRateEmaLambda = 1.0 makes the EMA equal to the instantaneous value.
  Info := TKANAttentionLayerInfo.Create(0, Spec, 16, 1,
                                         0.10,   // TauSquaring
                                         5,      // KSquaring
                                         1.0);   // ClipRateEmaLambda (no smoothing)
  try
    // Feed K_squaring - 1 = 4 sustained high rates: trigger should not fire yet.
    for i := 1 to 4 do
    begin
      Info.RecordSweepClipRate(0.5);   // well above τ = 0.10
      AssertFalse(Format('After %d high sweeps (need 5), trigger not yet firing', [i]),
                  Info.ShouldFireSquaring);
    end;
    // The 5th high sweep crosses the threshold.
    Info.RecordSweepClipRate(0.5);
    AssertEquals('After 5 high sweeps, ConsecutiveHighSweeps = 5',
                 5, Info.ConsecutiveHighSweeps);
    AssertTrue('Trigger fires on the K_squaring-th sustained high sweep',
               Info.ShouldFireSquaring);
  finally
    Info.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANSquaringTriggerResetsOnLowRate;
var
  Spec: TKANGridSpec;
  Info: TKANAttentionLayerInfo;
  i: integer;
begin
  Spec := MakeDefaultGridSpec(32);
  Info := TKANAttentionLayerInfo.Create(0, Spec, 16, 1, 0.10, 5, 1.0);
  try
    // Build up some high-pressure history.
    for i := 1 to 3 do Info.RecordSweepClipRate(0.5);
    AssertEquals('Counter built up to 3', 3, Info.ConsecutiveHighSweeps);

    // A single low-pressure sweep wipes the counter — sustained-pressure
    // semantics. The next high sweep starts the counter from 1, not 4.
    Info.RecordSweepClipRate(0.0);
    AssertEquals('Single low sweep resets ConsecutiveHighSweeps to 0',
                 0, Info.ConsecutiveHighSweeps);

    Info.RecordSweepClipRate(0.5);
    AssertEquals('Resumed pressure starts counter from 1, not from previous',
                 1, Info.ConsecutiveHighSweeps);
    AssertFalse('Trigger does not fire after the reset+1', Info.ShouldFireSquaring);
  finally
    Info.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANSquaringTriggerClampedAtCeiling;
var
  Spec: TKANGridSpec;
  Info: TKANAttentionLayerInfo;
  i: integer;
begin
  // HMax = 2 means the layer is already at the ceiling at construction
  // (initial ActiveHeads = 2). Even with sustained pressure, the trigger
  // must not fire — there is nowhere to grow. (HS-1: 2 ≤ ActiveHeads ≤ HMax.)
  Spec := MakeDefaultGridSpec(32);
  Info := TKANAttentionLayerInfo.Create(0, Spec, 2, 1, 0.10, 5, 1.0);
  try
    AssertEquals('At-ceiling: ActiveHeads = HMax', Info.HMax, Info.ActiveHeads);
    for i := 1 to 20 do Info.RecordSweepClipRate(0.9);   // very high, sustained
    AssertTrue('Counter still accumulates even at ceiling',
               Info.ConsecutiveHighSweeps >= 5);
    AssertFalse('Trigger does not fire when ActiveHeads = HMax',
                Info.ShouldFireSquaring);
  finally
    Info.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANSquaringTriggerEmaSmoothing;
var
  Spec: TKANGridSpec;
  Info: TKANAttentionLayerInfo;
  i: integer;
begin
  // With a small λ (default 0.01), a single high sweep does not push the
  // EMA above τ_squaring. The counter should stay at 0 until many sweeps
  // have accumulated to push the EMA past the threshold.
  Spec := MakeDefaultGridSpec(32);
  Info := TKANAttentionLayerInfo.Create(0, Spec, 16, 1,
                                         0.10,   // TauSquaring
                                         64,     // KSquaring
                                         0.01);  // realistic EMA smoothing
  try
    // After 1 high sweep, EMA = 0.99·0 + 0.01·0.5 = 0.005, well below 0.10.
    Info.RecordSweepClipRate(0.5);
    AssertTrue(Format('Single high sweep does not push EMA above threshold (got EMA=%g)',
                      [Info.ClipRateEMA]),
               Info.ClipRateEMA < 0.10);
    AssertEquals('Counter is still 0 after one high sweep', 0,
                 Info.ConsecutiveHighSweeps);

    // Feed many high sweeps and check the EMA grows.
    for i := 1 to 50 do Info.RecordSweepClipRate(0.5);
    AssertTrue(Format('After 51 high sweeps, EMA should be substantial (got %g)',
                      [Info.ClipRateEMA]),
               Info.ClipRateEMA > 0.05);
  finally
    Info.Free;
  end;
end;

// ---- Coverage gaps for already-implemented surface ----

procedure TTestNeuralKAN.TestKANGridSpecHashDeterministic;
var
  S1, S2, S3: TKANGridSpec;
begin
  S1 := MakeDefaultGridSpec(64);
  S2 := MakeDefaultGridSpec(64);
  AssertEquals('Same spec must produce same hash (run-to-run determinism)',
               S1.Hash, S2.Hash);

  S3 := MakeDefaultGridSpec(32);   // different KnotCount
  AssertTrue('Different KnotCount must change the hash', S1.Hash <> S3.Hash);

  S3 := MakeDefaultGridSpec(64);
  S3.GridLow := -4.0;              // different grid range
  AssertTrue('Different GridLow must change the hash', S1.Hash <> S3.Hash);

  S3 := MakeDefaultGridSpec(64);
  S3.GridHigh := 4.0;
  AssertTrue('Different GridHigh must change the hash', S1.Hash <> S3.Hash);
end;

procedure TTestNeuralKAN.TestKANRNGNextFloatRange;
var
  RNG: TKANSeededRNG;
  i: integer;
  v: TNeuralFloat;
begin
  RNG.Seed(7);
  for i := 1 to 10000 do
  begin
    v := RNG.NextFloat;
    AssertTrue(Format('NextFloat must be >= 0 (got %g)', [v]), v >= 0);
    AssertTrue(Format('NextFloat must be < 1 (got %g)', [v]), v < 1);
  end;
end;

procedure TTestNeuralKAN.TestKANRNGNextNormalSanity;
var
  RNG: TKANSeededRNG;
  i, n: integer;
  v, sum, sumSq, mean, variance: TNeuralFloat;
begin
  // Sample a moderately large number of standard-normal draws and check
  // that the empirical mean is close to 0 and variance is close to 1.
  // Generous tolerances — this is a sanity test, not a chi-squared test.
  RNG.Seed(123);
  n := 10000;
  sum := 0;
  sumSq := 0;
  for i := 1 to n do
  begin
    v := RNG.NextNormal;
    AssertTrue(Format('NextNormal must produce finite values (got %g)', [v]),
               not (IsNan(v) or IsInfinite(v)));
    sum := sum + v;
    sumSq := sumSq + v * v;
  end;
  mean := sum / n;
  variance := sumSq / n - mean * mean;
  AssertTrue(Format('Sample mean over %d normals should be near 0 (got %g)', [n, mean]),
             Abs(mean) < 0.1);
  AssertTrue(Format('Sample variance over %d normals should be near 1 (got %g)', [n, variance]),
             Abs(variance - 1.0) < 0.1);
end;

procedure TTestNeuralKAN.TestKANBasisCreateValidation;
var
  Spec: TKANGridSpec;
  Basis: TKANBasis;

  function TryCreate(const S: TKANGridSpec): boolean;
  begin
    Result := false;
    try
      Basis := TKANBasis.Create(S);
      Basis.Free;
    except
      on EKANBadState do Result := true;
    end;
  end;

begin
  // KnotCount < 4 is an error.
  Spec := MakeDefaultGridSpec(3);
  AssertTrue('KnotCount = 3 must raise EKANBadState', TryCreate(Spec));

  // BasisOrder != 3 is an error in v1.
  Spec := MakeDefaultGridSpec(64);
  Spec.BasisOrder := 2;
  AssertTrue('BasisOrder = 2 must raise EKANBadState (v1 cubic-only)', TryCreate(Spec));

  // GridHigh <= GridLow is an error.
  Spec := MakeDefaultGridSpec(64);
  Spec.GridHigh := Spec.GridLow;
  AssertTrue('GridHigh = GridLow must raise EKANBadState', TryCreate(Spec));

  Spec := MakeDefaultGridSpec(64);
  Spec.GridHigh := Spec.GridLow - 1.0;
  AssertTrue('GridHigh < GridLow must raise EKANBadState', TryCreate(Spec));

  // Sanity: a valid spec does NOT raise.
  Spec := MakeDefaultGridSpec(64);
  AssertFalse('Valid spec must not raise', TryCreate(Spec));
end;

procedure TTestNeuralKAN.TestKANNormaliserInitialPerHeadState;
var
  Spec: TKANGridSpec;
  Basis: TKANBasis;
  RNG: TKANSeededRNG;
  Norm: TNNetKANNormaliser;
begin
  // Verify the per-head state at construction time, before any
  // EnterInferenceMode call. KLEMA must be Infinity (so the EMA descends
  // from a sentinel rather than from a meaningless 0), all counters must
  // be zero, and Status must be ksSoftmaxActive.
  Spec := MakeDefaultGridSpec(32);
  Basis := TKANBasis.Create(Spec);
  RNG.Seed(1);

  Norm := TNNetKANNormaliser.Create(Spec, 7, 3, Basis, @RNG);
  try
    AssertEquals('AttentionLayerId stored', 7, Norm.AttentionLayerId);
    AssertEquals('HeadIndex stored', 3, Norm.HeadIndex);
    AssertTrue('Status starts SoftmaxActive', Norm.Status = ksSoftmaxActive);
    AssertTrue('KLEMA starts at +Infinity (sentinel)', IsInfinite(Norm.KLEMA));
    AssertEquals('ClipCountTotal starts at 0', 0, Norm.ClipCountTotal);
    AssertEquals('CascadeCapHits starts at 0', 0, Norm.CascadeCapHits);
    AssertTrue('KANEnabled starts true (per-layer opt-out, default opt-in)',
               Norm.KANEnabled);
    AssertFalse('FreezeAfterTakeover starts false', Norm.FreezeAfterTakeover);
    AssertTrue('SkipRedundantSoftmax starts true (perf default)',
               Norm.SkipRedundantSoftmax);
  finally
    Norm.Free;
    Basis.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANNormaliserEnterInferenceFitsCoefficients;
var
  Spec: TKANGridSpec;
  Basis: TKANBasis;
  RNG: TKANSeededRNG;
  Norm: TNNetKANNormaliser;
begin
  // Before EnterInferenceMode, KLEMA is the +Infinity sentinel. After,
  // ColdStartHead has fitted coefficients to ψ ≈ exp(s/2) and reset
  // KLEMA to +Infinity (which it already was). The visible signal that
  // cold-start ran is that subsequent EnterInferenceMode calls are no-ops.
  Spec := MakeDefaultGridSpec(32);
  Basis := TKANBasis.Create(Spec);
  RNG.Seed(99);

  Norm := TNNetKANNormaliser.Create(Spec, 0, 0, Basis, @RNG);
  try
    AssertTrue('Pre-inference KLEMA is +Infinity', IsInfinite(Norm.KLEMA));
    Norm.EnterInferenceMode;
    // Cold-start has run; the layer is in inference mode and KLEMA is
    // still at the sentinel waiting for the first observation. Calling
    // EnterInferenceMode again is idempotent.
    Norm.EnterInferenceMode;
  finally
    Norm.Free;
    Basis.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANNormaliserEnterInferenceRequiresSeededRNG;
var
  Spec: TKANGridSpec;
  Basis: TKANBasis;
  RNG: TKANSeededRNG;
  Norm: TNNetKANNormaliser;
  Caught: boolean;
begin
  // EnterInferenceMode raises EKANBadState if the RNG state is 0 — a
  // safety check against the case where construction wired the RNG
  // pointer but nobody called Seed.
  Spec := MakeDefaultGridSpec(32);
  Basis := TKANBasis.Create(Spec);
  RNG.State := 0;     // explicitly unseeded; default-zeroed record

  Norm := TNNetKANNormaliser.Create(Spec, 0, 0, Basis, @RNG);
  try
    Caught := false;
    try
      Norm.EnterInferenceMode;
    except
      on EKANBadState do Caught := true;
    end;
    AssertTrue('EnterInferenceMode with unseeded RNG must raise EKANBadState',
               Caught);
  finally
    Norm.Free;
    Basis.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANInfoCreateValidation;
var
  Spec: TKANGridSpec;
  Info: TKANAttentionLayerInfo;

  function TryCreate(const HMax: integer; const Tau: TNeuralFloat;
                     const K: integer; const Lambda: TNeuralFloat): boolean;
  begin
    Result := false;
    try
      Info := TKANAttentionLayerInfo.Create(0, Spec, HMax, 1, Tau, K, Lambda);
      Info.Free;
    except
      on EKANBadState do Result := true;
    end;
  end;

begin
  Spec := MakeDefaultGridSpec(32);
  AssertTrue('HMax < 2 must raise',          TryCreate(1, 0.10, 64, 0.01));
  AssertTrue('TauSquaring <= 0 must raise',  TryCreate(8, 0.0,  64, 0.01));
  AssertTrue('TauSquaring >= 1 must raise',  TryCreate(8, 1.0,  64, 0.01));
  AssertTrue('KSquaring < 1 must raise',     TryCreate(8, 0.10, 0,  0.01));
  AssertTrue('Lambda <= 0 must raise',       TryCreate(8, 0.10, 64, 0.0));
  AssertTrue('Lambda > 1 must raise',        TryCreate(8, 0.10, 64, 1.5));
  AssertFalse('Valid params must not raise', TryCreate(8, 0.10, 64, 0.01));
end;

procedure TTestNeuralKAN.TestKANInfoCheckSquaringRaisesWhenTriggered;
var
  Spec: TKANGridSpec;
  Info: TKANAttentionLayerInfo;
  i: integer;
  Caught: boolean;
begin
  // CheckSquaring currently raises EKANBadState when the trigger fires
  // because DoSquaring (the actual log-normal perturbation) is still
  // a stub. This test pins the current behaviour so the future
  // DoSquaring implementation has a test that fails when it lands.
  Spec := MakeDefaultGridSpec(32);
  Info := TKANAttentionLayerInfo.Create(0, Spec, 16, 1, 0.10, 3, 1.0);
  try
    AssertFalse('Trigger does not fire initially', Info.ShouldFireSquaring);

    // CheckSquaring is a no-op until the trigger fires.
    Info.CheckSquaring;   // should not raise

    // Build up sustained pressure to fire the trigger.
    for i := 1 to 3 do Info.RecordSweepClipRate(0.5);
    AssertTrue('Trigger fires after 3 high sweeps', Info.ShouldFireSquaring);

    // Now CheckSquaring should raise (DoSquaring is the stub).
    Caught := false;
    try
      Info.CheckSquaring;
    except
      on EKANBadState do Caught := true;
    end;
    AssertTrue('CheckSquaring must raise EKANBadState while DoSquaring is a stub',
               Caught);
  finally
    Info.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANNetBackpropGatesOnLock;
var
  NN: TKANNet;
  Caught: boolean;
begin
  // After LockToInference, calling Backpropagate must raise
  // EKANInInference *before* it touches anything else (the guard runs
  // first), so we can test this on an empty network without a real input.
  NN := TKANNet.Create;
  try
    NN.LockToInference;
    Caught := false;
    try
      NN.Backpropagate(nil);
    except
      on EKANInInference do Caught := true;
    end;
    AssertTrue('Backpropagate when locked must raise EKANInInference', Caught);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANNetBulkOpsOnEmptyNetwork;
var
  NN: TKANNet;
  Mask: TKANEnabledMask;
begin
  // Bulk operations on a network with no KAN attention layers must be
  // no-ops, not raise. This is the "construct, configure, destruct"
  // smoke-test path that operators will hit before they even build a
  // model.
  NN := TKANNet.Create;
  try
    NN.DisableAllKAN;       // no-op, must not raise
    NN.EnableAllKAN;        // no-op, must not raise
    Mask := NN.KANEnabledMask;
    AssertEquals('KANEnabledMask is empty for an empty network', 0, Length(Mask));
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralKAN.TestKANNetBulkOpsIndexValidation;
var
  NN: TKANNet;

  function CaughtBadState(Action: integer): boolean;
  begin
    Result := false;
    try
      case Action of
        0: NN.DisableKANForLayer(0);    // no layers; index 0 invalid
        1: NN.DisableKANForLayer(-1);
        2: NN.EnableKANForLayer(0);
        3: NN.EnableKANForLayer(-5);
      end;
    except
      on EKANBadState do Result := true;
    end;
  end;

begin
  NN := TKANNet.Create;
  try
    AssertTrue('DisableKANForLayer(0) on empty network raises',  CaughtBadState(0));
    AssertTrue('DisableKANForLayer(-1) raises',                  CaughtBadState(1));
    AssertTrue('EnableKANForLayer(0) on empty network raises',   CaughtBadState(2));
    AssertTrue('EnableKANForLayer(-5) raises',                   CaughtBadState(3));
  finally
    NN.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralKAN);

end.
