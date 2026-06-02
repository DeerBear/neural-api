unit neuralmmapdataset;

{$mode objfpc}{$H+}

(*
neuralmmapdataset
=================

Cross-platform, memory-mapped text dataset loader for line-oriented corpora
(e.g. TinyStories), independent of any particular network architecture.

WHAT IT DOES
  Maps the dataset file once, read-only, and indexes every kept line's byte
  offset in a single forward pass. Worker threads then read sample text
  directly from the single shared mapping -- there is no per-thread copy of
  the corpus and no eager load of the whole file into a TStringList. This is
  the "load once, work downstream" data source: the file is mapped once and
  shared; the per-sample encoding work happens downstream of it.

WHY IT HELPS
  A TStringList-backed loader holds the entire corpus in heap memory and, with
  data-parallel training, the access pattern fans out across worker threads.
  A read-only shared mapping keeps a single copy backed by the OS page cache;
  pages are faulted in on demand, which both lowers resident memory and lets
  the OS overlap I/O with compute when threads outnumber cores.

INDEPENDENCE
  Depends only on the core units neuralvolume and neuralnetwork (to size the
  sample volumes from a bound network's first / last layers). It pulls in no
  optimiser, attention, or other optional machinery, so it can be dropped into
  any FitLoading-based example unchanged.

PLATFORMS
  * Unix (Linux / macOS / BSD): mmap via BaseUnix.
  * Windows:                    CreateFileMapping / MapViewOfFile.
  The mapping is read-only; extracted lines are copied out (lowercased and
  sentinel-appended) only when a sample is built, so the mapping itself is
  never mutated. Line-offset arithmetic goes through PtrUInt / Int64 so it is
  safe past 2 GB on 64-bit builds.

ENCODING
  Matches the stock SimpleNLP char-level convention: each kept line is
  lowercased and a sentinel byte (#1) is appended; the input is the reversed
  one-hot encoding of a prefix, the target is the next character as a softmax
  class. Lines shorter than MinSampleSize are skipped during indexing.

USAGE
    var
      Mmap: TNNetMappedTextDataset;
    ...
      Mmap := TNNetMappedTextDataset.Create({MinSampleSize=}3);
      Mmap.LoadDataset('datasets/tinystories.txt');
      Mmap.BindNetwork(NN);            // call after the network is built
      NFit.FitLoading(
        NN, Mmap.Count, ValidationCount, 0, BatchSize,
        @Mmap.GetTrainingPair,         // signatures match TNeuralDataLoadingFit
        @Mmap.GetValidationPair,
        @Mmap.GetValidationPair);
    ...
      Mmap.Free;

NOTE ON RANDOMNESS
  GetTrainingPair draws samples with the global Random, exactly as the stock
  TStringList-backed SimpleNLP getter does; GetValidationPair is deterministic
  in Idx so validation is reproducible.
*)

interface

uses
  {$IFDEF UNIX}
  BaseUnix,
  {$ENDIF}
  {$IFDEF WINDOWS}
  Windows,
  {$ENDIF}
  Classes, SysUtils, Math,
  neuralvolume, neuralnetwork;

const
  csMmapDefaultMinSampleSize = 3;
  csMmapSentinel = #1;
  // Progress cadence for the one-pass line index, otherwise a multi-GB file
  // scans silently. Report roughly every this-many bytes.
  csMmapReportBytes = 256 * 1024 * 1024;

type

  { TNNetMappedTextDataset }

  TNNetMappedTextDataset = class
  private
    {$IFDEF WINDOWS}
    FFile: THandle;
    FMap: THandle;
    {$ENDIF}
    {$IFDEF UNIX}
    FFd: cint;
    {$ENDIF}
    FView: PByte;
    FSize: Int64;
    FLineOfs: array of Int64;    // byte offset of each kept line's first char
    FLineLen: array of integer;  // byte length, excluding the line break
    FCount: integer;
    FNN: TNNet;
    FMinSampleSize: integer;
    FMaxPredictCharPos: integer; // 0 = no clamp on the training prefix length
    function ByteAt(const Ofs: Int64): byte; inline;
    function ExtractLine(const Idx: integer): string;
    procedure MapFile(const AFileName: string);
    procedure UnmapFile;
  public
    constructor Create(const AMinSampleSize: integer = csMmapDefaultMinSampleSize;
      const AMaxPredictCharPos: integer = 0);
    destructor Destroy; override;
    // Maps AFileName and indexes its line offsets in one pass.
    procedure LoadDataset(const AFileName: string);
    // Must be called before Get*Pair so volumes can be sized from the network.
    procedure BindNetwork(ANN: TNNet);
    // FitLoading-compatible getters. Idx/ThreadId are accepted for signature
    // compatibility; training ignores Idx (random sampling), validation uses
    // it for deterministic, reproducible coverage.
    procedure GetTrainingPair(Idx: integer; ThreadId: integer;
      pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer;
      pInput, pOutput: TNNetVolume);
    // Number of kept lines (>= MinSampleSize). Use as the training count.
    property Count: integer read FCount;
    // Optional clamp on the training prefix length (mirrors the stock example's
    // FMaxPredictCharPos). 0 disables it. Validation is never clamped.
    property MaxPredictCharPos: integer
      read FMaxPredictCharPos write FMaxPredictCharPos;
  end;

implementation

constructor TNNetMappedTextDataset.Create(const AMinSampleSize: integer;
  const AMaxPredictCharPos: integer);
begin
  inherited Create;
  {$IFDEF WINDOWS}
  FFile := INVALID_HANDLE_VALUE;
  FMap := 0;
  {$ENDIF}
  {$IFDEF UNIX}
  FFd := -1;
  {$ENDIF}
  FView := nil;
  FSize := 0;
  FCount := 0;
  FNN := nil;
  if AMinSampleSize < 1 then
    FMinSampleSize := 1
  else
    FMinSampleSize := AMinSampleSize;
  FMaxPredictCharPos := AMaxPredictCharPos;
end;

destructor TNNetMappedTextDataset.Destroy;
begin
  UnmapFile;
  inherited Destroy;
end;

procedure TNNetMappedTextDataset.MapFile(const AFileName: string);
{$IFDEF WINDOWS}
var
  SizeHi, SizeLo: DWORD;
begin
  FFile := CreateFile(PChar(AFileName), GENERIC_READ, FILE_SHARE_READ, nil,
    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
  if FFile = INVALID_HANDLE_VALUE then
    raise Exception.CreateFmt(
      'TNNetMappedTextDataset: cannot open %s (err %d)',
      [AFileName, GetLastError]);
  SizeHi := 0;
  SizeLo := GetFileSize(FFile, @SizeHi);
  FSize := (Int64(SizeHi) shl 32) or SizeLo;
  if FSize <= 0 then
    raise Exception.CreateFmt('TNNetMappedTextDataset: %s is empty', [AFileName]);
  FMap := CreateFileMapping(FFile, nil, PAGE_READONLY, 0, 0, nil);
  if FMap = 0 then
    raise Exception.CreateFmt(
      'TNNetMappedTextDataset: CreateFileMapping failed on %s (err %d)',
      [AFileName, GetLastError]);
  FView := MapViewOfFile(FMap, FILE_MAP_READ, 0, 0, 0);
  if FView = nil then
    raise Exception.CreateFmt(
      'TNNetMappedTextDataset: MapViewOfFile failed on %s (err %d)',
      [AFileName, GetLastError]);
end;
{$ENDIF}
{$IFDEF UNIX}
begin
  FFd := FpOpen(AFileName, O_RDONLY);
  if FFd < 0 then
    raise Exception.CreateFmt(
      'TNNetMappedTextDataset: cannot open %s (errno %d)',
      [AFileName, fpgeterrno]);
  FSize := FpLseek(FFd, 0, SEEK_END);
  FpLseek(FFd, 0, SEEK_SET);
  if FSize <= 0 then
  begin
    FpClose(FFd);
    FFd := -1;
    raise Exception.CreateFmt('TNNetMappedTextDataset: %s is empty', [AFileName]);
  end;
  FView := PByte(Fpmmap(nil, FSize, PROT_READ, MAP_PRIVATE, FFd, 0));
  if (FView = nil) or (Pointer(FView) = Pointer(-1)) then
  begin
    FpClose(FFd);
    FFd := -1;
    FView := nil;
    raise Exception.CreateFmt(
      'TNNetMappedTextDataset: mmap failed on %s (errno %d)',
      [AFileName, fpgeterrno]);
  end;
end;
{$ENDIF}

procedure TNNetMappedTextDataset.UnmapFile;
begin
  {$IFDEF WINDOWS}
  if FView <> nil then UnmapViewOfFile(FView);
  if FMap <> 0 then CloseHandle(FMap);
  if FFile <> INVALID_HANDLE_VALUE then CloseHandle(FFile);
  FFile := INVALID_HANDLE_VALUE;
  FMap := 0;
  {$ENDIF}
  {$IFDEF UNIX}
  if (FView <> nil) and (FSize > 0) then Fpmunmap(FView, FSize);
  if FFd >= 0 then FpClose(FFd);
  FFd := -1;
  {$ENDIF}
  FView := nil;
end;

function TNNetMappedTextDataset.ByteAt(const Ofs: Int64): byte;
begin
  Result := PByte(PtrUInt(FView) + PtrUInt(Ofs))^;
end;

procedure TNNetMappedTextDataset.LoadDataset(const AFileName: string);
var
  P, LineStart, NextReport: Int64;
  LineByteLen, Capacity, NewCap: integer;

  procedure AddLine(const Ofs: Int64; const Len: integer);
  begin
    if Len >= FMinSampleSize then
    begin
      if FCount >= Capacity then
      begin
        if Capacity = 0 then NewCap := 1 shl 20 else NewCap := Capacity * 2;
        SetLength(FLineOfs, NewCap);
        SetLength(FLineLen, NewCap);
        Capacity := NewCap;
      end;
      FLineOfs[FCount] := Ofs;
      FLineLen[FCount] := Len;
      Inc(FCount);
    end;
  end;

begin
  MapFile(AFileName);

  // One forward pass: split on LF, trim a trailing CR -- ReadLn semantics.
  // Periodic progress, since this scans the whole (multi-GB) file and would
  // otherwise be a silent gap before the first training log.
  WriteLn(Format('  memory-mapped %d MB; indexing lines...',
    [FSize div (1024 * 1024)]));
  Flush(Output);
  Capacity := 0;
  FCount := 0;
  LineStart := 0;
  NextReport := csMmapReportBytes;
  P := 0;
  while P < FSize do
  begin
    if ByteAt(P) = 10 then            // LF
    begin
      LineByteLen := Integer(P - LineStart);
      if (LineByteLen > 0) and (ByteAt(P - 1) = 13) then
        Dec(LineByteLen);             // strip CR
      AddLine(LineStart, LineByteLen);
      LineStart := P + 1;
    end;
    if P >= NextReport then
    begin
      WriteLn(Format('  mmap indexing: %d MB scanned, %d lines kept',
        [P div (1024 * 1024), FCount]));
      Flush(Output);
      NextReport := NextReport + csMmapReportBytes;
    end;
    Inc(P);
  end;
  if LineStart < FSize then           // trailing line with no final newline
    AddLine(LineStart, Integer(FSize - LineStart));

  WriteLn(Format('  mmap index complete: %d lines kept from %d MB',
    [FCount, FSize div (1024 * 1024)]));
  Flush(Output);
end;

function TNNetMappedTextDataset.ExtractLine(const Idx: integer): string;
var
  Raw: AnsiString;
  Len: integer;
begin
  Len := FLineLen[Idx];
  SetLength(Raw, Len);
  if Len > 0 then
    Move(PByte(PtrUInt(FView) + PtrUInt(FLineOfs[Idx]))^, Raw[1], Len);
  // Lowercase + sentinel, matching the stock SimpleNLP loader.
  Result := LowerCase(string(Raw)) + csMmapSentinel;
end;

procedure TNNetMappedTextDataset.BindNetwork(ANN: TNNet);
begin
  FNN := ANN;
end;

procedure TNNetMappedTextDataset.GetTrainingPair(Idx: integer; ThreadId: integer;
  pInput, pOutput: TNNetVolume);
var
  Sample: string;
  SampleLen, CutPos, TokInt: integer;
begin
  if FNN.GetFirstLayer().Output.Size <> pInput.Size then
    pInput.ReSize(FNN.GetFirstLayer().Output);
  if FNN.GetLastLayer().Output.Size <> pOutput.Size then
    pOutput.ReSize(FNN.GetLastLayer().Output);

  Sample := ExtractLine(Random(FCount));
  SampleLen := Min(Length(Sample), pInput.SizeX);
  if FMaxPredictCharPos > 0 then
    SampleLen := Min(FMaxPredictCharPos, SampleLen);

  if SampleLen <= FMinSampleSize then
    CutPos := FMinSampleSize
  else
    CutPos := Random(SampleLen - FMinSampleSize) + FMinSampleSize;

  TokInt := Min(Ord(Sample[CutPos + 1]), pInput.Depth - 1);
  pInput.OneHotEncodingReversed(copy(Sample, 1, CutPos));
  pOutput.SetClassForSoftMax(TokInt);
  pOutput.Tag := TokInt;
end;

procedure TNNetMappedTextDataset.GetValidationPair(Idx: integer; ThreadId: integer;
  pInput, pOutput: TNNetVolume);
var
  Sample: string;
  SampleId, SampleLen, CutPos, TokInt: integer;
begin
  if FNN.GetFirstLayer().Output.Size <> pInput.Size then
    pInput.ReSize(FNN.GetFirstLayer().Output);
  if FNN.GetLastLayer().Output.Size <> pOutput.Size then
    pOutput.ReSize(FNN.GetLastLayer().Output);

  SampleId := Idx mod FCount;                 // deterministic, wrapped
  Sample := ExtractLine(SampleId);
  SampleLen := Min(Length(Sample), pInput.SizeX);

  CutPos := (Idx mod (1 + SampleLen - FMinSampleSize))
            + FMinSampleSize - 1;

  TokInt := Min(Ord(Sample[CutPos + 1]), pInput.Depth - 1);
  pInput.OneHotEncodingReversed(copy(Sample, 1, CutPos));
  pOutput.SetClassForSoftMax(TokInt);
  pOutput.Tag := TokInt;
end;

end.
