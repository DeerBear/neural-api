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
  Result := 0;
  raise EKANBadState.Create('TKANBasis.BasisSquaredSum: not implemented');
end;

procedure TKANBasis.FitPsiToExp(out Coeffs: array of TNeuralFloat);
begin
  // TODO: least-squares fit ψ(s) ≈ exp(s/2) on a dense grid;
  // resulting coefficients give φ = ψ² ≈ exp(s) over [GridLow, GridHigh].
  raise EKANBadState.Create('TKANBasis.FitPsiToExp: not implemented');
end;

end.
