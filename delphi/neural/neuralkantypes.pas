(*
neuralkantypes
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

unit neuralkantypes;

(*
KAN attention foundational types: exceptions, enums, records, RNG.

STANDALONE DELPHI PORT (modern Delphi 10.x - 12 Athens).
Hand-converted from the FPC/Lazarus source under ../../neural/. No FPC
conditionals: this tree is meant to be opened and built natively in Delphi.
This unit has no dependency on any other KAN unit and on no neural-api
unit other than neuralvolume (for TNeuralFloat).

Mechanism specification:    docs/kan_attention_spec.md
Implementation decisions:   docs/kan_implementation_pascal.md §3
*)

interface

uses
  Classes, SysUtils, Math,
  neuralvolume;

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
  //  ENABLED-MASK RETURN TYPE
  // ===================================================================

  /// Per-attention-layer KAN enable state, returned by TKANNet.KANEnabledMask.
  TKANEnabledMask = array of boolean;

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

implementation

// =====================================================================
//  TKANGridSpec
// =====================================================================

function TKANGridSpec.Hash: UInt64;
begin
  // SplitMix64-style mix of all four fields. Stable across runs.
  // Disable overflow / range checking: the multiplications are intentional
  // unsigned wraparound, and Trunc() of a negative Single is cast bit-wise
  // into UInt64. Delphi's default {$Q+}/{$R+} would raise EIntOverflow /
  // ERangeError on these otherwise.
  {$Q-}{$R-}
  Result := UInt64(KnotCount) * UInt64($9E3779B97F4A7C15);
  Result := Result xor (UInt64(BasisOrder) * UInt64($BF58476D1CE4E5B9));
  Result := Result xor (UInt64(Trunc(GridLow * 1000)) * UInt64($94D049BB133111EB));
  Result := Result xor (UInt64(Trunc(GridHigh * 1000)) * UInt64($D6E8FEB86659FD93));
  {$Q+}{$R+}
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
  // SplitMix64 relies on modular UInt64 arithmetic — the add and the two
  // mixing multiplies are expected to wrap. Disable Delphi's default
  // {$Q+}/{$R+} for the routine body.
  {$Q-}{$R-}
  State := State + UInt64($9E3779B97F4A7C15);
  Result := State;
  Result := (Result xor (Result shr 30)) * UInt64($BF58476D1CE4E5B9);
  Result := (Result xor (Result shr 27)) * UInt64($94D049BB133111EB);
  Result := Result xor (Result shr 31);
  {$Q+}{$R+}
end;

function TKANSeededRNG.NextFloat: TNeuralFloat;
begin
  // 53-bit-equivalent uniform [0, 1). The explicit TNeuralFloat() cast on
  // the UInt64 operand is required for Delphi, which does not implicitly
  // convert UInt64 to a floating-point type.
  Result := TNeuralFloat(NextU64 shr 11) / TNeuralFloat($1FFFFFFFFFFFFF);
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

end.
