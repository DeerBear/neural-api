unit kantransformerdata;

(*
TinyStories character-level dataset loader and pair-getter source for the
KAN transformer example.

Shared between training and inference programs: training uses all three
pair-getters; inference uses GetValidationPair only (for CalibrateAlpha).

Streaming loader. Replaces TStringList.LoadFromFile + two backwards-
traversal passes with a single forward pass: read a line, apply min-
length filter, lowercase + append sentinel, add to the list, print
progress every 100k lines. Constant working memory during load (one line
buffered at a time, plus the growing FDataset), one pass instead of
three, and visible progress so the user can see it's alive.
*)

interface

uses
  Classes, SysUtils, Math,
  neuralvolume, neuralnetwork;

const
  csMinSampleSize = 3;  // Minimum of 3 characters per sample.

type
  TKANTransformerDataset = class
  private
    FDataset: TStringList;
    FDatasetSize: integer;
    FFileName: string;
    FNN: TNNet;
    FMaxPredictCharPos: integer;
    FContextLen: integer;
    // Length-distribution accumulators populated during LoadDataset.
    // Used to derive a data-driven context length via log-log statistics
    // (robust to heavy tails: economists use the same trick to compare
    // wildly different unit scales, e.g. GDP vs population).
    FLengthSum, FLengthSumSq: double;
    FLogSum, FLogSumSq: double;
    FLogLogSum, FLogLogSumSq: double;
    FMinLen, FMaxLen: integer;
    function GetMeanLen: double;
    function GetMeanLogLen: double;
    function GetStdLogLen: double;
    function GetMeanLogLogLen: double;
    function GetStdLogLogLen: double;
    function GetRecommendedContextLen: integer;
  public
    constructor Create(const AFileName: string; AContextLen: integer);
    destructor Destroy; override;
    procedure LoadDataset;
    // Bind the dataset to a network so the pair-getters can size pInput/
    // pOutput from the network's first/last layers. Must be called before
    // any pair-getter is used (typically right after the network is built).
    procedure BindNetwork(ANN: TNNet);
    procedure GetTrainingPair(Idx: integer; ThreadId: integer;
      pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer;
      pInput, pOutput: TNNetVolume);
    procedure GetTestPair(Idx: integer; ThreadId: integer;
      pInput, pOutput: TNNetVolume);
    property DatasetSize: integer read FDatasetSize;
    // Source file path, so a memory-mapped training source can be built from
    // the same corpus the eager loader read.
    property FileName: string read FFileName;
    // Curriculum window for training samples: pair-getter caps the sample
    // length by this value. OnAfterEpoch typically grows/shrinks this in
    // response to training accuracy.
    property MaxPredictCharPos: integer
      read FMaxPredictCharPos write FMaxPredictCharPos;
    // Length-distribution diagnostics. All populated after LoadDataset.
    property MinLen: integer read FMinLen;
    property MaxLen: integer read FMaxLen;
    property MeanLen: double read GetMeanLen;
    // Single-log mean: Exp(mean(ln(len))) = geometric mean. Robust if
    // length distribution is log-normal.
    property MeanLogLen: double read GetMeanLogLen;
    property StdLogLen: double read GetStdLogLen;
    // Double-log mean: Exp(Exp(mean(ln(ln(len))))). Robust even when the
    // single-log distribution still has a heavy tail (Pareto-like). The
    // central tendency in log-log space is invariant to outliers that
    // would yank single-log statistics around.
    property MeanLogLogLen: double read GetMeanLogLogLen;
    property StdLogLogLen: double read GetStdLogLogLen;
    // Data-driven context size from the log-log mean. Use this as the
    // input to BuildKANTransformer1M unless overridden.
    property RecommendedContextLen: integer read GetRecommendedContextLen;
  end;

implementation

constructor TKANTransformerDataset.Create(const AFileName: string;
  AContextLen: integer);
begin
  inherited Create;
  FFileName := AFileName;
  FContextLen := AContextLen;
  FDataset := TStringList.Create;
  FDatasetSize := 0;
  FMaxPredictCharPos := AContextLen;
  FNN := nil;
  FLengthSum := 0;
  FLengthSumSq := 0;
  FLogSum := 0;
  FLogSumSq := 0;
  FLogLogSum := 0;
  FLogLogSumSq := 0;
  FMinLen := MaxInt;
  FMaxLen := 0;
end;

destructor TKANTransformerDataset.Destroy;
begin
  FDataset.Free;
  inherited Destroy;
end;

procedure TKANTransformerDataset.LoadDataset;
var
  Reader: TStreamReader;
  Line: string;
  LineNum, LineLen: integer;
  LogLen, LogLogLen: double;
begin
  WriteLn('Streaming dataset from ', FFileName, '...');
  Flush(Output);
  // Pre-size to avoid ~20 reallocation+copy cycles as the list grows.
  // 2.5M is an over-estimate for TinyStories (~2M stories); over-sizing
  // costs ~10MB of unused pointer slots, undersizing costs reallocations.
  FDataset.Capacity := 2500000;
  Reader := TStreamReader.Create(FFileName, TEncoding.UTF8);
  try
    LineNum := 0;
    while not Reader.EndOfStream do
    begin
      Line := Reader.ReadLine;
      LineLen := Length(Line);
      if LineLen >= csMinSampleSize then
      begin
        FDataset.Add(LowerCase(Line) + chr(1));
        // Distribution accumulators -- raw, single-log, double-log.
        // Guard the double-log: Ln(Ln(x)) requires Ln(x) > 0, so x > e
        // (~2.72). csMinSampleSize=3 satisfies this, but the guard
        // keeps the code robust if csMinSampleSize is lowered.
        FLengthSum := FLengthSum + LineLen;
        FLengthSumSq := FLengthSumSq + LineLen * LineLen;
        LogLen := Ln(LineLen);
        FLogSum := FLogSum + LogLen;
        FLogSumSq := FLogSumSq + LogLen * LogLen;
        if LogLen > 0 then
        begin
          LogLogLen := Ln(LogLen);
          FLogLogSum := FLogLogSum + LogLogLen;
          FLogLogSumSq := FLogLogSumSq + LogLogLen * LogLogLen;
        end;
        if LineLen < FMinLen then FMinLen := LineLen;
        if LineLen > FMaxLen then FMaxLen := LineLen;
      end;
      Inc(LineNum);
      if (LineNum mod 100000) = 0 then
      begin
        WriteLn('  read ', LineNum, ' lines, kept ', FDataset.Count);
        Flush(Output);
      end;
    end;
  finally
    Reader.Free;
  end;
  FDatasetSize := FDataset.Count;
  WriteLn('Loaded dataset with ', FDatasetSize, ' rows');
  if FDatasetSize > 0 then
  begin
    WriteLn(Format(
      '  Length distribution: min=%d, max=%d, arith mean=%.1f',
      [FMinLen, FMaxLen, GetMeanLen]));
    WriteLn(Format(
      '  Log-space    : mean=%.3f (Exp -> %.1f), stddev=%.3f',
      [GetMeanLogLen, Exp(GetMeanLogLen), GetStdLogLen]));
    WriteLn(Format(
      '  Log-log space: mean=%.3f (Exp(Exp(.)) -> %.1f), stddev=%.3f',
      [GetMeanLogLogLen, Exp(Exp(GetMeanLogLogLen)), GetStdLogLogLen]));
    // Regime classifier. MeanLogLogLen in 0.x means typical Ln(length)
    // sits in (1, e), i.e. typical length is in (e, e^e) ~ (2.7, 15.2),
    // so the second log compresses nothing meaningful. Above 1 it does.
    if GetMeanLogLogLen < 1.0 then
      WriteLn('  Log-log regime: OVERKILL (mean<1; single log would suffice)')
    else if GetMeanLogLogLen < 1.5 then
      WriteLn('  Log-log regime: MARGINAL (mean in [1.0, 1.5); single log is close)')
    else
      WriteLn('  Log-log regime: JUSTIFIED (mean>=1.5; second log materially compresses)');
    WriteLn(Format(
      '  Recommended context length (from log-log mean): %d',
      [GetRecommendedContextLen]));
  end;
  Flush(Output);
end;

function TKANTransformerDataset.GetMeanLen: double;
begin
  if FDatasetSize > 0 then
    Result := FLengthSum / FDatasetSize
  else
    Result := 0;
end;

function TKANTransformerDataset.GetMeanLogLen: double;
begin
  if FDatasetSize > 0 then
    Result := FLogSum / FDatasetSize
  else
    Result := 0;
end;

function TKANTransformerDataset.GetStdLogLen: double;
var
  M, V: double;
begin
  if FDatasetSize > 0 then
  begin
    M := FLogSum / FDatasetSize;
    V := FLogSumSq / FDatasetSize - M * M;
    if V > 0 then Result := Sqrt(V) else Result := 0;
  end
  else
    Result := 0;
end;

function TKANTransformerDataset.GetMeanLogLogLen: double;
begin
  if FDatasetSize > 0 then
    Result := FLogLogSum / FDatasetSize
  else
    Result := 0;
end;

function TKANTransformerDataset.GetStdLogLogLen: double;
var
  M, V: double;
begin
  if FDatasetSize > 0 then
  begin
    M := FLogLogSum / FDatasetSize;
    V := FLogLogSumSq / FDatasetSize - M * M;
    if V > 0 then Result := Sqrt(V) else Result := 0;
  end
  else
    Result := 0;
end;

function TKANTransformerDataset.GetRecommendedContextLen: integer;
begin
  // Default sizing rule: Exp(Exp(mean(Ln(Ln(len))))). Log-log central
  // tendency is robust to power-law tails that would dominate single-log
  // statistics. Caller can override by passing a different ContextLen
  // to BuildKANTransformer1M.
  if FDatasetSize > 0 then
    Result := Round(Exp(Exp(GetMeanLogLogLen)))
  else
    Result := FContextLen;
end;

procedure TKANTransformerDataset.BindNetwork(ANN: TNNet);
begin
  FNN := ANN;
end;

procedure TKANTransformerDataset.GetTrainingPair(Idx: integer; ThreadId: integer;
  pInput, pOutput: TNNetVolume);
var
  SampleId: integer;
  SampleLen: integer;
  SampleCutPosition: integer;
  ExpectedTokenChar: char;
  ExpectedTokenInt: integer;
begin
  if FNN.GetFirstLayer().Output.Size <> pInput.Size then
    pInput.ReSize(FNN.GetFirstLayer().Output);
  if FNN.GetLastLayer().Output.Size <> pOutput.Size then
    pOutput.ReSize(FNN.GetLastLayer().Output);
  SampleId := Random(FDatasetSize);
  SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
  SampleLen := Min(FMaxPredictCharPos, SampleLen);
  SampleCutPosition := Random(SampleLen - csMinSampleSize) + csMinSampleSize;
  ExpectedTokenChar := FDataset[SampleId][SampleCutPosition + 1];
  ExpectedTokenInt := Min(Ord(ExpectedTokenChar), pInput.Depth - 1);
  pInput.OneHotEncodingReversed(copy(FDataset[SampleId], 1, SampleCutPosition));
  pOutput.SetClassForSoftMax(ExpectedTokenInt);
  pOutput.Tag := ExpectedTokenInt;
end;

procedure TKANTransformerDataset.GetValidationPair(Idx: integer; ThreadId: integer;
  pInput, pOutput: TNNetVolume);
var
  SampleId: integer;
  SampleLen: integer;
  SampleCutPosition: integer;
  ExpectedTokenChar: char;
  ExpectedTokenInt: integer;
begin
  if FNN.GetFirstLayer().Output.Size <> pInput.Size then
    pInput.ReSize(FNN.GetFirstLayer().Output);
  if FNN.GetLastLayer().Output.Size <> pOutput.Size then
    pOutput.ReSize(FNN.GetLastLayer().Output);
  SampleId := Idx;
  SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
  SampleCutPosition := (Idx mod (1 + SampleLen - csMinSampleSize)) + csMinSampleSize - 1;
  ExpectedTokenChar := FDataset[SampleId][SampleCutPosition + 1];
  ExpectedTokenInt := Min(Ord(ExpectedTokenChar), pInput.Depth - 1);
  pInput.OneHotEncodingReversed(copy(FDataset[SampleId], 1, SampleCutPosition));
  pOutput.SetClassForSoftMax(ExpectedTokenInt);
  pOutput.Tag := ExpectedTokenInt;
end;

procedure TKANTransformerDataset.GetTestPair(Idx: integer; ThreadId: integer;
  pInput, pOutput: TNNetVolume);
begin
  GetValidationPair(Idx, ThreadId, pInput, pOutput);
end;

end.
