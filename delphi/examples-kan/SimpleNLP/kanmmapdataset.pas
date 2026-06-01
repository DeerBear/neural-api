unit kanmmapdataset;

(*
Memory-mapped TinyStories dataset for the KAN transformer.

Maps the dataset file once, read-only, and indexes line offsets in a single
pass. Every worker thread reads sample text directly from the single shared
mapping -- no per-thread copy of the corpus, no eager load into a TStringList.
This is the "load once, work downstream" data source: the loading is shared,
the per-sample work is downstream of it.

Windows memory mapping (CreateFileMapping / MapViewOfFile). The mapped view is
read-only; extracted lines are copied out (and lowercased + sentinel-appended)
only when a sample is built, so the mapping itself is never mutated. Pairs
directly with kanprefetch: BuildTrainingSample is a drop-in TKANBuildSampleProc.

STATUS: v1, written without a Delphi toolchain available. Validate on a real
run. Pointer arithmetic is done through NativeUInt so it does not depend on
{$POINTERMATH}.
*)

interface

uses
  Winapi.Windows, Classes, SysUtils, Math,
  neuralvolume, neuralnetwork;

const
  csMmapMinSampleSize = 3;
  csMmapSentinel = #1;
  // Progress cadence for the one-pass line index (otherwise silent on a
  // multi-GB file). Report every this-many bytes scanned.
  csMmapReportBytes = 256 * 1024 * 1024;

type
  TKANMappedDataset = class
  private
    FFile: THandle;
    FMap: THandle;
    FView: PByte;
    FSize: Int64;
    FLineOfs: array of Int64;    // byte offset of each kept line's first char
    FLineLen: array of integer;  // length in bytes, excluding the line break
    FCount: integer;
    FNN: TNNet;
    FContextLen: integer;
    function ByteAt(const Ofs: Int64): byte; inline;
    function ExtractLine(const Idx: integer): string;
  public
    constructor Create(const AContextLen: integer);
    destructor Destroy; override;
    procedure LoadDataset(const AFileName: string);
    // Must be called before BuildTrainingSample so the volumes can be sized
    // from the network's first/last layers.
    procedure BindNetwork(ANN: TNNet);
    // Builds one training sample from a random mapped line. Same encoding as
    // the eager loader's getter. Drop-in for kanprefetch's TKANBuildSampleProc.
    procedure BuildTrainingSample(Input, Output: TNNetVolume);
    property Count: integer read FCount;
  end;

implementation

constructor TKANMappedDataset.Create(const AContextLen: integer);
begin
  inherited Create;
  FFile := INVALID_HANDLE_VALUE;
  FMap := 0;
  FView := nil;
  FSize := 0;
  FCount := 0;
  FContextLen := AContextLen;
  FNN := nil;
end;

destructor TKANMappedDataset.Destroy;
begin
  if FView <> nil then UnmapViewOfFile(FView);
  if FMap <> 0 then CloseHandle(FMap);
  if FFile <> INVALID_HANDLE_VALUE then CloseHandle(FFile);
  inherited Destroy;
end;

function TKANMappedDataset.ByteAt(const Ofs: Int64): byte;
begin
  Result := PByte(NativeUInt(FView) + NativeUInt(Ofs))^;
end;

procedure TKANMappedDataset.LoadDataset(const AFileName: string);
var
  SizeHi, SizeLo: DWORD;
  P, LineStart, NextReport: Int64;
  LineByteLen, Capacity, NewCap: integer;

  procedure AddLine(const Ofs: Int64; const Len: integer);
  begin
    if Len >= csMmapMinSampleSize then
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
  FFile := CreateFile(PChar(AFileName), GENERIC_READ, FILE_SHARE_READ, nil,
    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
  if FFile = INVALID_HANDLE_VALUE then
    raise Exception.CreateFmt('TKANMappedDataset: cannot open %s (err %d)',
      [AFileName, GetLastError]);

  SizeLo := GetFileSize(FFile, @SizeHi);
  FSize := (Int64(SizeHi) shl 32) or SizeLo;

  FMap := CreateFileMapping(FFile, nil, PAGE_READONLY, 0, 0, nil);
  if FMap = 0 then
    raise Exception.CreateFmt('TKANMappedDataset: CreateFileMapping failed (err %d)',
      [GetLastError]);
  FView := MapViewOfFile(FMap, FILE_MAP_READ, 0, 0, 0);
  if FView = nil then
    raise Exception.CreateFmt('TKANMappedDataset: MapViewOfFile failed (err %d)',
      [GetLastError]);

  // One pass: split on LF, trim a trailing CR -- mirrors ReadLine semantics.
  // Periodic progress, since this scans the whole (multi-GB) file and would
  // otherwise be a silent black box between "Computing..." and the first
  // training log. ByteAt is inline + NativeUInt-safe past 2 GB.
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
  if LineStart < FSize then           // trailing line, no final newline
    AddLine(LineStart, Integer(FSize - LineStart));
  WriteLn(Format('  mmap index complete: %d lines kept from %d MB',
    [FCount, FSize div (1024 * 1024)]));
  Flush(Output);
end;

function TKANMappedDataset.ExtractLine(const Idx: integer): string;
var
  Raw: AnsiString;
  Len: integer;
begin
  Len := FLineLen[Idx];
  SetLength(Raw, Len);
  if Len > 0 then
    Move(PByte(NativeUInt(FView) + NativeUInt(FLineOfs[Idx]))^, Raw[1], Len);
  // Lowercase + sentinel, matching the eager loader.
  Result := LowerCase(string(Raw)) + csMmapSentinel;
end;

procedure TKANMappedDataset.BindNetwork(ANN: TNNet);
begin
  FNN := ANN;
end;

procedure TKANMappedDataset.BuildTrainingSample(Input, Output: TNNetVolume);
var
  Sample: string;
  SampleLen, CutPos, TokInt: integer;
begin
  if FNN.GetFirstLayer().Output.Size <> Input.Size then
    Input.ReSize(FNN.GetFirstLayer().Output);
  if FNN.GetLastLayer().Output.Size <> Output.Size then
    Output.ReSize(FNN.GetLastLayer().Output);
  Sample := ExtractLine(Random(FCount));
  SampleLen := Min(Length(Sample), Input.SizeX);
  CutPos := Random(SampleLen - csMmapMinSampleSize) + csMmapMinSampleSize;
  TokInt := Min(Ord(Sample[CutPos + 1]), Input.Depth - 1);
  Input.OneHotEncodingReversed(copy(Sample, 1, CutPos));
  Output.SetClassForSoftMax(TokInt);
  Output.Tag := TokInt;
end;

end.
