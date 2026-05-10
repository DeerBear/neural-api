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

initialization
  RegisterTest(TTestNeuralKAN);

end.
