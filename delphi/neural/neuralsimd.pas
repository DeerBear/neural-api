(*
neuralsimd
Copyright (C) 2016 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

unit neuralsimd;

(*
STANDALONE DELPHI PORT — SIMD math kernels (first functional group).

Hand-converted from the FPC/Lazarus neuralvolume.pas (the {$IFDEF AVX64}
asm blocks + their documented scalar semantics). No FPC conditionals.

WHY THIS UNIT EXISTS
  In the FPC source the AVX* kernels and the canonical float types both
  live inside neuralvolume.pas, guarded by {$IFDEF AVX32/AVX64} with NO
  pure-Pascal twin (the upstream Delphi path simply does not compile this
  unit). To get a clean, buildable Delphi port the kernels are isolated
  here as the lowest layer: neuralvolume USES this unit and its hundreds
  of AVX* call sites resolve unchanged (signatures are byte-identical to
  the FPC versions, including overloads and parameter names — so the
  calling convention the rest of the library expects does not change).

SIMD STRATEGY (agreed)
  * Default build: pure Pascal. Always compiled, correct, portable
    (Win32/Win64), no asm. This is what ships unless you opt in.
  * Optional build: define NEURAL_AVX (project conditional) on Win64 to
    use the hand-ported Delphi BASM (AVX2/FMA path).

  IMPORTANT: the {$IFDEF NEURAL_AVX} assembly was converted by hand and
  has NOT been compiled or run (no Delphi/x64 toolchain was available
  during the port). It is a faithful transcription of the upstream
  AVX64 path: FPC clobber-lists removed (clobbered RAX/RCX/RDX + ymm0..5
  are Win64-volatile; the ptrC mul-add kernel additionally preserves the
  non-volatile RBX), AVX512/AVX2 compile-time branches resolved to the
  concrete AVX2/FMA form, the Pascal scaffolding kept identical so symbol
  references behave as upstream. VALIDATE on real hardware (numeric
  parity vs the Pascal path) before enabling NEURAL_AVX in production.
*)

interface

type
  TNeuralFloat = Single;
  TNeuralFloatDynArr = array of TNeuralFloat;
  TNeuralFloatPtr = ^TNeuralFloat;
  TNeuralFloat4 = array[0..3] of TNeuralFloat;
  {$IFDEF CPUX86}
  TNeuralFloatArr = array[0..1024*2048] of TNeuralFloat;
  {$ELSE}
  // Win64: cap below 2GB to stay inside a single static-array index range.
  TNeuralFloatArr = array[0..Maxint div SizeOf(TNeuralFloat) div 8] of TNeuralFloat;
  {$ENDIF}
  TNeuralFloatArrPtr = ^TNeuralFloatArr;
  TNeuralIntegerArray = array of integer;

const
  csMinAvxSize = 16;

// PtrA[i] := FillOp
procedure AVXFill(PtrA: TNeuralFloatArrPtr; FillOp: TNeuralFloat; NumElements: integer);
// PtrA[i] := PtrA[i] + MulOp*PtrB[i]
procedure AVXMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; MulOp: TNeuralFloat; NumElements: integer); overload;
// PtrA[i] := PtrA[i] + PtrB[i]*PtrC[i]
procedure AVXMulAdd(PtrA, PtrB, PtrC: TNeuralFloatArrPtr; NumElements: integer); overload;
// PtrA[i] := PtrA[i]*MulOp1 + MulOp2*PtrB[i]
procedure AVXMulMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; MulOp1, MulOp2: TNeuralFloat; NumElements: integer);
// PtrA[i] := Max(0, PtrB[i])
procedure AVXCopyRelu(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
// PtrA[i] := PtrA[i]*MulOp
procedure AVXMul(PtrA: TNeuralFloatArrPtr; MulOp: TNeuralFloat; NumElements: integer); overload;
// PtrA[i] := PtrA[i]*PtrB[i]
procedure AVXMul(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer); overload;
// PtrA[i] := PtrA[i] + PtrB[i]
procedure AVXAdd(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
// PtrA[i] := PtrA[i] - PtrB[i]
procedure AVXSub(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
// Result := Sum |PtrA[i] - PtrB[i]|
function  AVXSumDiff(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
// Result := Sum (PtrA[i] - PtrB[i])^2
function  AVXDistanceSqr(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
// Result := Sum PtrA[i]
function  AVXGetSum(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single;
// Result := Sum PtrA[i]^2
function  AVXGetSumSqr(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single;
// Result := Sum PtrA[i]*PtrB[i]
function  AVXDotProduct(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;

implementation

uses
  Math;

// =====================================================================
//  AVXFill
// =====================================================================
procedure AVXFill(PtrA: TNeuralFloatArrPtr; FillOp: TNeuralFloat; NumElements: integer);
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  FillOpPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    FillOpPtr := Addr(FillOp);
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, FillOpPtr
  VBROADCASTSS ymm0, [rdx]
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  vmovups [rax],    ymm0
  vmovups [rax+32], ymm0
  vmovups [rax+64], ymm0
  vmovups [rax+96], ymm0
  add rax, 128
  dec ecx
  jnz @LargeAddLoop
@SkipLargeAddLoop:
  vzeroupper
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  movups [rax], xmm0
  add rax, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  end;
  end;
  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := FillOp;
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := FillOp;
      if MissedElements>2 then PtrA^[localNumElements+2] := FillOp;
    end;
  end;
end;
{$ELSE}
var
  I: integer;
begin
  for I := 0 to NumElements - 1 do
    PtrA^[I] := FillOp;
end;
{$ENDIF}

// =====================================================================
//  AVXMulAdd (scalar):  PtrA += MulOp * PtrB
// =====================================================================
procedure AVXMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; MulOp: TNeuralFloat; NumElements: integer);
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  MulOpPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    MulOpPtr := Addr(MulOp);
  asm
  mov ecx, localNumElements
  mov rax, PtrB
  mov rdx, MulOpPtr
  VBROADCASTSS ymm5, [rdx]
  mov rdx, PtrA
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  vmovups ymm0, [rdx]
  vmovups ymm1, [rdx+32]
  vmovups ymm2, [rdx+64]
  vmovups ymm3, [rdx+96]
  vfmadd231ps ymm0, ymm5, [rax]
  vfmadd231ps ymm1, ymm5, [rax+32]
  vfmadd231ps ymm2, ymm5, [rax+64]
  vfmadd231ps ymm3, ymm5, [rax+96]
  vmovups [rdx],    ymm0
  vmovups [rdx+32], ymm1
  vmovups [rdx+64], ymm2
  vmovups [rdx+96], ymm3
  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop
@SkipLargeAddLoop:
  vzeroupper
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  movups  xmm2, [rax]
  movups  xmm4, [rdx]
  mulps   xmm2, xmm5
  addps   xmm4, xmm2
  movups  [rdx], xmm4
  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  end;
  end;
  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements] + MulOp*PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1] + MulOp*PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2] + MulOp*PtrB^[localNumElements+2];
    end;
  end;
end;
{$ELSE}
var
  I: integer;
begin
  for I := 0 to NumElements - 1 do
    PtrA^[I] := PtrA^[I] + MulOp*PtrB^[I];
end;
{$ENDIF}

// =====================================================================
//  AVXMulAdd (elementwise):  PtrA += PtrB * PtrC
//  (upstream asm_avx64_mulladd_ptra_ptrb_ptrc_num macro; uses RBX which
//   is Win64-non-volatile, so it is explicitly preserved here.)
// =====================================================================
procedure AVXMulAdd(PtrA, PtrB, PtrC: TNeuralFloatArrPtr; NumElements: integer);
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rdx, PtrA
  mov rax, PtrB
  push rbx
  mov rbx, PtrC
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  vmovups ymm4, [rbx]
  vmovups ymm5, [rbx+32]
  vmulps  ymm0, ymm4, [rax]
  vmulps  ymm1, ymm5, [rax+32]
  vaddps  ymm0, ymm0, [rdx]
  vaddps  ymm1, ymm1, [rdx+32]
  vmovups [rdx],    ymm0
  vmovups [rdx+32], ymm1
  vmovups ymm4, [rbx+64]
  vmovups ymm5, [rbx+96]
  vmulps  ymm2, ymm4, [rax+64]
  vmulps  ymm3, ymm5, [rax+96]
  vaddps  ymm2, ymm2, [rdx+64]
  vaddps  ymm3, ymm3, [rdx+96]
  vmovups [rdx+64], ymm2
  vmovups [rdx+96], ymm3
  add rax, 128
  add rdx, 128
  add rbx, 128
  dec ecx
  jnz @LargeAddLoop
@SkipLargeAddLoop:
  vzeroupper
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  movups  xmm2, [rax]
  movups  xmm5, [rbx]
  movups  xmm4, [rdx]
  mulps   xmm2, xmm5
  addps   xmm4, xmm2
  movups  [rdx], xmm4
  add rax, 16
  add rbx, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  pop rbx
  end;
  end;
  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements] + PtrB^[localNumElements]*PtrC^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1] + PtrB^[localNumElements+1]*PtrC^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2] + PtrB^[localNumElements+2]*PtrC^[localNumElements+2];
    end;
  end;
end;
{$ELSE}
var
  I: integer;
begin
  for I := 0 to NumElements - 1 do
    PtrA^[I] := PtrA^[I] + PtrB^[I]*PtrC^[I];
end;
{$ENDIF}

// =====================================================================
//  AVXMulMulAdd:  PtrA := PtrA*MulOp1 + MulOp2*PtrB
// =====================================================================
procedure AVXMulMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; MulOp1, MulOp2: TNeuralFloat; NumElements: integer);
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  MulOpPtr1, MulOpPtr2: pointer;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    MulOpPtr1 := Addr(MulOp1);
    MulOpPtr2 := Addr(MulOp2);
  asm
  mov ecx, localNumElements
  mov rax, PtrB
  mov rdx, MulOpPtr1
  VBROADCASTSS ymm5, [rdx]
  mov rdx, MulOpPtr2
  VBROADCASTSS ymm4, [rdx]
  mov rdx, PtrA
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  vmulps  ymm0, ymm4, [rax]
  vmulps  ymm1, ymm4, [rax+32]
  vmulps  ymm2, ymm5, [rdx]
  vmulps  ymm3, ymm5, [rdx+32]
  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm1, ymm1, ymm3
  vmovups [rdx],    ymm0
  vmovups [rdx+32], ymm1
  vmulps  ymm0, ymm4, [rax+64]
  vmulps  ymm1, ymm4, [rax+96]
  vmulps  ymm2, ymm5, [rdx+64]
  vmulps  ymm3, ymm5, [rdx+96]
  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm1, ymm1, ymm3
  vmovups [rdx+64], ymm0
  vmovups [rdx+96], ymm1
  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop
@SkipLargeAddLoop:
  vzeroupper
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  movups  xmm2, [rax]
  movups  xmm1, [rdx]
  mulps   xmm2, xmm4
  mulps   xmm1, xmm5
  addps   xmm1, xmm2
  movups  [rdx], xmm1
  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  end;
  end;
  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements]*MulOp1 + MulOp2*PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1]*MulOp1 + MulOp2*PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2]*MulOp1 + MulOp2*PtrB^[localNumElements+2];
    end;
  end;
end;
{$ELSE}
var
  I: integer;
begin
  for I := 0 to NumElements - 1 do
    PtrA^[I] := PtrA^[I]*MulOp1 + MulOp2*PtrB^[I];
end;
{$ENDIF}

// =====================================================================
//  AVXCopyRelu:  PtrA := Max(0, PtrB)
// =====================================================================
procedure AVXCopyRelu(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  ZeroVar: TNeuralFloat;
  ZeroVarPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  ZeroVar := 0;
  if localNumElements > 0 then
  begin
    ZeroVarPtr := Addr(ZeroVar);
  asm
  mov ecx, localNumElements
  mov rax, PtrB
  mov rdx, ZeroVarPtr
  VBROADCASTSS ymm5, [rdx]
  mov rdx, PtrA
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  VMAXPS ymm0, ymm5, [rax]
  VMAXPS ymm1, ymm5, [rax+32]
  VMAXPS ymm2, ymm5, [rax+64]
  VMAXPS ymm3, ymm5, [rax+96]
  vmovups [rdx],    ymm0
  vmovups [rdx+32], ymm1
  vmovups [rdx+64], ymm2
  vmovups [rdx+96], ymm3
  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop
@SkipLargeAddLoop:
  vzeroupper
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  movups  xmm2, [rax]
  MAXPS   xmm2, xmm5
  movups  [rdx], xmm2
  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  end;
  end;
  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := Max(0,PtrB^[localNumElements]);
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := Max(0,PtrB^[localNumElements+1]);
      if MissedElements>2 then PtrA^[localNumElements+2] := Max(0,PtrB^[localNumElements+2]);
    end;
  end;
end;
{$ELSE}
var
  I: integer;
begin
  for I := 0 to NumElements - 1 do
    if PtrB^[I] > 0 then PtrA^[I] := PtrB^[I] else PtrA^[I] := 0;
end;
{$ENDIF}

// =====================================================================
//  AVXMul (scalar):  PtrA := PtrA * MulOp
// =====================================================================
procedure AVXMul(PtrA: TNeuralFloatArrPtr; MulOp: TNeuralFloat; NumElements: integer);
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  MulOpPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    MulOpPtr := Addr(MulOp);
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, MulOpPtr
  VBROADCASTSS ymm0, [rdx]
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  vmulps  ymm2, ymm0, [rax]
  vmulps  ymm3, ymm0, [rax+32]
  vmulps  ymm4, ymm0, [rax+64]
  vmulps  ymm5, ymm0, [rax+96]
  vmovups [rax],    ymm2
  vmovups [rax+32], ymm3
  vmovups [rax+64], ymm4
  vmovups [rax+96], ymm5
  add rax, 128
  dec ecx
  jnz @LargeAddLoop
@SkipLargeAddLoop:
  vzeroupper
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  movups  xmm2, [rax]
  mulps   xmm2, xmm0
  movups [rax], xmm2
  add rax, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  end;
  end;
  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements] * MulOp;
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1] * MulOp;
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2] * MulOp;
    end;
  end;
end;
{$ELSE}
var
  I: integer;
begin
  for I := 0 to NumElements - 1 do
    PtrA^[I] := PtrA^[I] * MulOp;
end;
{$ENDIF}

// =====================================================================
//  AVXMul (elementwise):  PtrA := PtrA * PtrB
// =====================================================================
procedure AVXMul(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrB
  mov rdx, PtrA
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  vmovups  ymm0, [rax]
  vmovups  ymm1, [rax+32]
  vmovups  ymm2, [rax+64]
  vmovups  ymm3, [rax+96]
  vmulps  ymm0, ymm0, [rdx]
  vmulps  ymm1, ymm1, [rdx+32]
  vmulps  ymm2, ymm2, [rdx+64]
  vmulps  ymm3, ymm3, [rdx+96]
  vmovups [rdx],    ymm0
  vmovups [rdx+32], ymm1
  vmovups [rdx+64], ymm2
  vmovups [rdx+96], ymm3
  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop
@SkipLargeAddLoop:
  vzeroupper
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  movups  xmm2, [rax]
  movups  xmm4, [rdx]
  mulps   xmm2, xmm4
  movups  [rdx], xmm2
  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  end;
  end;
  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements] * PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1] * PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
    end;
  end;
end;
{$ELSE}
var
  I: integer;
begin
  for I := 0 to NumElements - 1 do
    PtrA^[I] := PtrA^[I] * PtrB^[I];
end;
{$ENDIF}

// =====================================================================
//  AVXAdd:  PtrA := PtrA + PtrB
// =====================================================================
procedure AVXAdd(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]
  vaddps  ymm2, ymm2, [rdx]
  vaddps  ymm3, ymm3, [rdx+32]
  vaddps  ymm4, ymm4, [rdx+64]
  vaddps  ymm5, ymm5, [rdx+96]
  vmovups [rax],    ymm2
  vmovups [rax+32], ymm3
  vmovups [rax+64], ymm4
  vmovups [rax+96], ymm5
  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop
  vzeroupper
@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  vzeroupper
  movups xmm2, [rax]
  movups xmm3, [rdx]
  addps xmm2, xmm3
  movups [rax], xmm2
  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  end;
  end;
  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements] + PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1] + PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2] + PtrB^[localNumElements+2];
    end;
  end;
end;
{$ELSE}
var
  I: integer;
begin
  for I := 0 to NumElements - 1 do
    PtrA^[I] := PtrA^[I] + PtrB^[I];
end;
{$ENDIF}

// =====================================================================
//  AVXSub:  PtrA := PtrA - PtrB
//  (Reconstructed from the AVXAdd skeleton with subps/vsubps — the
//   upstream AVXSub asm follows the identical structure. Pascal path
//   is authoritative; validate the asm before enabling NEURAL_AVX.)
// =====================================================================
procedure AVXSub(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]
  vsubps  ymm2, ymm2, [rdx]
  vsubps  ymm3, ymm3, [rdx+32]
  vsubps  ymm4, ymm4, [rdx+64]
  vsubps  ymm5, ymm5, [rdx+96]
  vmovups [rax],    ymm2
  vmovups [rax+32], ymm3
  vmovups [rax+64], ymm4
  vmovups [rax+96], ymm5
  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop
  vzeroupper
@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  vzeroupper
  movups xmm2, [rax]
  movups xmm3, [rdx]
  subps xmm2, xmm3
  movups [rax], xmm2
  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  end;
  end;
  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements] - PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1] - PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2] - PtrB^[localNumElements+2];
    end;
  end;
end;
{$ELSE}
var
  I: integer;
begin
  for I := 0 to NumElements - 1 do
    PtrA^[I] := PtrA^[I] - PtrB^[I];
end;
{$ENDIF}

// =====================================================================
//  AVXSumDiff:  Result := Sum |PtrA - PtrB|
// =====================================================================
function AVXSumDiff(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB
  vxorps ymm0, ymm0, ymm0
  VPCMPEQD  ymm1, ymm1, ymm1
  VPSRLD    ymm1, ymm1, 1
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]
  vsubps  ymm2, ymm2, [rdx]
  vsubps  ymm3, ymm3, [rdx+32]
  vsubps  ymm4, ymm4, [rdx+64]
  vsubps  ymm5, ymm5, [rdx+96]
  vandps  ymm2, ymm2, ymm1
  vandps  ymm3, ymm3, ymm1
  vandps  ymm4, ymm4, ymm1
  vandps  ymm5, ymm5, ymm1
  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm0, ymm0, ymm3
  vaddps  ymm0, ymm0, ymm4
  vaddps  ymm0, ymm0, ymm5
  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop
  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2
@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  vzeroupper
  movups xmm2, [rax]
  movups xmm3, [rdx]
  subps  xmm2, xmm3
  andps  xmm2, xmm1
  addps  xmm0, xmm2
  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  vzeroupper
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0
  movups vRes, xmm0
  end;
    Result := vRes[0];
  end else
    Result := 0;
  if MissedElements>0 then
  begin
    if MissedElements = 1 then
      Result := Result + Abs(PtrA^[localNumElements]-PtrB^[localNumElements])
    else if MissedElements = 2 then
      Result := Result +
        Abs(PtrA^[localNumElements]-PtrB^[localNumElements]) +
        Abs(PtrA^[localNumElements+1]-PtrB^[localNumElements+1])
    else
      Result := Result +
        Abs(PtrA^[localNumElements]-PtrB^[localNumElements]) +
        Abs(PtrA^[localNumElements+1]-PtrB^[localNumElements+1]) +
        Abs(PtrA^[localNumElements+2]-PtrB^[localNumElements+2]);
  end;
end;
{$ELSE}
var
  I: integer;
begin
  Result := 0;
  for I := 0 to NumElements - 1 do
    Result := Result + Abs(PtrA^[I] - PtrB^[I]);
end;
{$ENDIF}

// =====================================================================
//  AVXDistanceSqr:  Result := Sum (PtrA - PtrB)^2
// =====================================================================
function AVXDistanceSqr(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB
  vxorps ymm0, ymm0, ymm0
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
@LargeAddLoop:
  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]
  vsubps  ymm2, ymm2, [rdx]
  vsubps  ymm3, ymm3, [rdx+32]
  vsubps  ymm4, ymm4, [rdx+64]
  vsubps  ymm5, ymm5, [rdx+96]
  vmulps  ymm2, ymm2, ymm2
  vmulps  ymm3, ymm3, ymm3
  vmulps  ymm4, ymm4, ymm4
  vmulps  ymm5, ymm5, ymm5
  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm0, ymm0, ymm3
  vaddps  ymm0, ymm0, ymm4
  vaddps  ymm0, ymm0, ymm5
  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop
  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2
@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  vzeroupper
  movups xmm2, [rax]
  movups xmm3, [rdx]
  subps  xmm2, xmm3
  mulps  xmm2, xmm2
  addps  xmm0, xmm2
  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  vzeroupper
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0
  movups vRes, xmm0
  end;
    Result := vRes[0];
  end else
    Result := 0;
  if MissedElements>0 then
  begin
    if MissedElements = 1 then
      Result := Result + Sqr(PtrA^[localNumElements]-PtrB^[localNumElements])
    else if MissedElements = 2 then
      Result := Result +
        Sqr(PtrA^[localNumElements]-PtrB^[localNumElements]) +
        Sqr(PtrA^[localNumElements+1]-PtrB^[localNumElements+1])
    else
      Result := Result +
        Sqr(PtrA^[localNumElements]-PtrB^[localNumElements]) +
        Sqr(PtrA^[localNumElements+1]-PtrB^[localNumElements+1]) +
        Sqr(PtrA^[localNumElements+2]-PtrB^[localNumElements+2]);
  end;
end;
{$ELSE}
var
  I: integer;
  d: TNeuralFloat;
begin
  Result := 0;
  for I := 0 to NumElements - 1 do
  begin
    d := PtrA^[I] - PtrB^[I];
    Result := Result + d*d;
  end;
end;
{$ENDIF}

// =====================================================================
//  AVXGetSum:  Result := Sum PtrA
//  (Reconstructed from the AVXDotProduct reduction skeleton without the
//   second-operand multiply. Pascal path is authoritative.)
// =====================================================================
function AVXGetSum(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single;
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  vxorps ymm0, ymm0, ymm0
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
  vxorps ymm1, ymm1, ymm1
@LargeAddLoop:
  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]
  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm1, ymm1, ymm3
  vaddps  ymm0, ymm0, ymm4
  vaddps  ymm1, ymm1, ymm5
  add rax, 128
  dec ecx
  jnz @LargeAddLoop
  vaddps ymm0, ymm0, ymm1
  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2
@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  vzeroupper
  movups xmm2, [rax]
  addps xmm0, xmm2
  add rax, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  vzeroupper
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0
  movups vRes, xmm0
  end;
    Result := vRes[0];
  end else
    Result := 0;
  if MissedElements>0 then
  begin
    if MissedElements = 1 then
      Result := Result + PtrA^[localNumElements]
    else if MissedElements = 2 then
      Result := Result + PtrA^[localNumElements] + PtrA^[localNumElements+1]
    else
      Result := Result + PtrA^[localNumElements] + PtrA^[localNumElements+1] + PtrA^[localNumElements+2];
  end;
end;
{$ELSE}
var
  I: integer;
begin
  Result := 0;
  for I := 0 to NumElements - 1 do
    Result := Result + PtrA^[I];
end;
{$ENDIF}

// =====================================================================
//  AVXGetSumSqr:  Result := Sum PtrA^2
//  (Reconstructed from the AVXDotProduct skeleton with PtrB = PtrA.
//   Pascal path is authoritative.)
// =====================================================================
function AVXGetSumSqr(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single;
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  vxorps ymm0, ymm0, ymm0
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
  vxorps ymm1, ymm1, ymm1
@LargeAddLoop:
  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]
  vfmadd231ps ymm0, ymm2, ymm2
  vfmadd231ps ymm1, ymm3, ymm3
  vfmadd231ps ymm0, ymm4, ymm4
  vfmadd231ps ymm1, ymm5, ymm5
  add rax, 128
  dec ecx
  jnz @LargeAddLoop
  vaddps ymm0, ymm0, ymm1
  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2
@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  vzeroupper
  movups xmm2, [rax]
  mulps xmm2, xmm2
  addps xmm0, xmm2
  add rax, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  vzeroupper
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0
  movups vRes, xmm0
  end;
    Result := vRes[0];
  end else
    Result := 0;
  if MissedElements>0 then
  begin
    if MissedElements = 1 then
      Result := Result + Sqr(PtrA^[localNumElements])
    else if MissedElements = 2 then
      Result := Result + Sqr(PtrA^[localNumElements]) + Sqr(PtrA^[localNumElements+1])
    else
      Result := Result + Sqr(PtrA^[localNumElements]) + Sqr(PtrA^[localNumElements+1]) + Sqr(PtrA^[localNumElements+2]);
  end;
end;
{$ELSE}
var
  I: integer;
begin
  Result := 0;
  for I := 0 to NumElements - 1 do
    Result := Result + PtrA^[I]*PtrA^[I];
end;
{$ENDIF}

// =====================================================================
//  AVXDotProduct:  Result := Sum PtrA*PtrB
// =====================================================================
function AVXDotProduct(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
{$IF Defined(NEURAL_AVX) and Defined(CPUX64)}
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB
  vxorps ymm0, ymm0, ymm0
  push rcx
  shr ecx,5
  jz @SkipLargeAddLoop
  vxorps ymm1, ymm1, ymm1
@LargeAddLoop:
  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]
  vfmadd231ps ymm0, ymm2, [rdx]
  vfmadd231ps ymm1, ymm3, [rdx+32]
  vfmadd231ps ymm0, ymm4, [rdx+64]
  vfmadd231ps ymm1, ymm5, [rdx+96]
  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop
  vaddps ymm0, ymm0, ymm1
  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2
@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2
@SmallAddLoop:
  vzeroupper
  movups xmm2, [rax]
  movups xmm3, [rdx]
  mulps xmm2, xmm3
  addps xmm0, xmm2
  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop
@EndAdd:
  vzeroupper
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0
  movups vRes, xmm0
  end;
    Result := vRes[0];
  end else
    Result := 0;
  if MissedElements>0 then
  begin
    if MissedElements = 1 then
      Result := Result + PtrA^[localNumElements] * PtrB^[localNumElements]
    else if MissedElements = 2 then
      Result := Result +
        PtrA^[localNumElements] * PtrB^[localNumElements] +
        PtrA^[localNumElements+1] * PtrB^[localNumElements+1]
    else
      Result := Result +
        PtrA^[localNumElements] * PtrB^[localNumElements] +
        PtrA^[localNumElements+1] * PtrB^[localNumElements+1] +
        PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
  end;
end;
{$ELSE}
var
  I: integer;
begin
  Result := 0;
  for I := 0 to NumElements - 1 do
    Result := Result + PtrA^[I] * PtrB^[I];
end;
{$ENDIF}

end.
