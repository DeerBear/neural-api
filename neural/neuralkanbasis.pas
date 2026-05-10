(*
neuralkanbasis
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

unit neuralkanbasis;

(*
KAN attention B-spline basis. One instance per attention layer;
shared across all H_max heads. Immutable after construction.

Mechanism specification:    docs/kan_attention_spec.md §5.1
Implementation decisions:   docs/kan_implementation_pascal.md §3.4

v1 status: SKELETON. Method bodies are stubs that raise EKANBadState.
*)

{$include neuralnetwork.inc}

interface

uses
  Classes, SysUtils, Math,
  neuralvolume,
  neuralkantypes;

type
  /// Immutable B-spline basis. Constructed from a TKANGridSpec; provides
  /// the local k+1 basis activations at any score s, the squared-basis
  /// sum used as the NLMS denominator, and the cold-start psi-space fit
  /// to exp(s/2) so that phi = psi^2 ~= exp(s) over the grid range.
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

    /// Cold-start fit: writes psi-coefficients into Coeffs such that
    /// psi(s)^2 ≈ exp(s) over the grid range. Least-squares on
    /// (s, exp(s/2)) pairs (spec §5.1, §8.4).
    procedure FitPsiToExp(out Coeffs: array of TNeuralFloat);

    property GridSpec: TKANGridSpec read FGridSpec;
  end;

implementation

// =====================================================================
//  TKANBasis  —  cardinal cubic B-spline on a uniform grid
// =====================================================================
//
//  Knots are evenly spaced across [GridLow, GridHigh]; one basis
//  function is centred at each knot. Knot spacing:
//
//      h := (GridHigh - GridLow) / (KnotCount - 1)
//      knot[i] := GridLow + i * h        for i in 0 .. KnotCount-1
//
//  For a query score s in [GridLow, GridHigh], the cardinal cubic
//  B-spline B_i(s) is evaluated in normalised form t := (s - knot[i]) / h:
//
//      B(t) = (1/6) * (4 - 6t² + 3|t|³)        for |t| < 1
//      B(t) = (1/6) * (2 - |t|)³               for 1 ≤ |t| < 2
//      B(t) = 0                                for |t| ≥ 2
//
//  Support is 4 knot intervals = k+1 active basis functions per
//  evaluation, matching spec §5.1 locality.
//
//  Boundary handling: basis functions whose index would fall outside
//  [0, KnotCount-1] contribute zero. Near the boundary the partition
//  of unity does not hold; downstream row-normalisation absorbs this.
// =====================================================================

constructor TKANBasis.Create(const Spec: TKANGridSpec);
var
  h: TNeuralFloat;
  i: integer;
begin
  inherited Create;

  if Spec.KnotCount < 4 then
    raise EKANBadState.CreateFmt(
      'TKANBasis.Create: KnotCount must be >= 4 (got %d)', [Spec.KnotCount]);
  if Spec.GridHigh <= Spec.GridLow then
    raise EKANBadState.Create(
      'TKANBasis.Create: GridHigh must be strictly greater than GridLow');
  if Spec.BasisOrder <> 3 then
    raise EKANBadState.CreateFmt(
      'TKANBasis.Create: only cubic basis (BasisOrder = 3) is supported in v1, got %d',
      [Spec.BasisOrder]);

  FGridSpec := Spec;

  h := (Spec.GridHigh - Spec.GridLow) / (Spec.KnotCount - 1);
  SetLength(FKnots, Spec.KnotCount);
  for i := 0 to Spec.KnotCount - 1 do
    FKnots[i] := Spec.GridLow + i * h;
end;

destructor TKANBasis.Destroy;
begin
  SetLength(FKnots, 0);
  inherited Destroy;
end;

procedure TKANBasis.Evaluate(const s: TNeuralFloat;
                              out FirstIdx: integer;
                              Vals: PSingle);
var
  h, t, absT, v: TNeuralFloat;
  i, idx: integer;
  ValsArr: PSingleArray absolute Vals;
begin
  h := (FGridSpec.GridHigh - FGridSpec.GridLow) / (FGridSpec.KnotCount - 1);
  // The 4 active basis functions are those centred at knot indices
  // floor((s - GridLow)/h) - 1, ..., +2.
  FirstIdx := Floor((s - FGridSpec.GridLow) / h) - 1;

  for i := 0 to 3 do
  begin
    idx := FirstIdx + i;
    if (idx < 0) or (idx >= FGridSpec.KnotCount) then
    begin
      ValsArr^[i] := 0;
      continue;
    end;
    t := (s - FKnots[idx]) / h;
    absT := Abs(t);
    if absT < 1 then
      v := (1.0 / 6.0) * (4 - 6 * t * t + 3 * absT * absT * absT)
    else if absT < 2 then
      v := (1.0 / 6.0) * Power(2 - absT, 3)
    else
      v := 0;
    ValsArr^[i] := v;
  end;
end;

function TKANBasis.BasisSquaredSum(const s: TNeuralFloat): TNeuralFloat;
var
  Vals: array[0..3] of TNeuralFloat;
  FirstIdx, i: integer;
begin
  Evaluate(s, FirstIdx, @Vals[0]);
  Result := 0;
  for i := 0 to 3 do Result := Result + Vals[i] * Vals[i];
end;

procedure TKANBasis.FitPsiToExp(out Coeffs: array of TNeuralFloat);
var
  i: integer;
begin
  // Spec §5.1 / §8.4: fit ψ(s) ≈ exp(s/2), so that φ = ψ² ≈ exp(s).
  //
  // v1 uses simple collocation: c_i := exp(knot[i] / 2). This is not
  // the optimal least-squares fit — the resulting ψ(s) at non-knot
  // points is a smoothed version of exp(s/2) due to the cubic basis
  // averaging — but it lands within ~1% of exp(s/2) at every knot
  // for typical knot spacing (h ≈ 0.25 nat for the default grid).
  //
  // Phase M's NLMS rule (spec §5.5.2) refines the fit toward exp(s)
  // exactly during shadow-mimicry, so the cold-start only needs to
  // start KL_ema descending. A future optimisation could solve the
  // tridiagonal collocation system for an exact knot-fit, or the
  // normal equations for a true least-squares fit on a dense grid.

  if Length(Coeffs) <> FGridSpec.KnotCount then
    raise EKANBadState.CreateFmt(
      'TKANBasis.FitPsiToExp: Coeffs length %d does not match KnotCount %d',
      [Length(Coeffs), FGridSpec.KnotCount]);

  for i := 0 to FGridSpec.KnotCount - 1 do
    Coeffs[i] := Exp(FKnots[i] / 2);
end;

end.
