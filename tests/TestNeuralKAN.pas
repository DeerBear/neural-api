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
begin
  Fail(NOT_IMPLEMENTED);
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
    Basis.Evaluate(s, FirstIdx, @Vals[0]);
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
      Basis.Evaluate(s, FirstIdx, @Vals[0]);
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
      Basis.Evaluate(s, FirstIdx, @Vals[0]);
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
      Basis.Evaluate(s, FirstIdx, @Vals[0]);
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

initialization
  RegisterTest(TTestNeuralKAN);

end.
