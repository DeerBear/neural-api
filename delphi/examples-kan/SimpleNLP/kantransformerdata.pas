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
    // Curriculum window for training samples: pair-getter caps the sample
    // length by this value. OnAfterEpoch typically grows/shrinks this in
    // response to training accuracy.
    property MaxPredictCharPos: integer
      read FMaxPredictCharPos write FMaxPredictCharPos;
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
  LineNum: integer;
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
      if Length(Line) >= csMinSampleSize then
        FDataset.Add(LowerCase(Line) + chr(1));
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
  Flush(Output);
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
