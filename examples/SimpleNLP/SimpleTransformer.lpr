program SimpleTransformer;
(*
Copyright (C) 2023 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  neuralnetwork,
  neuralvolume,
  neuralfit,
  neuraldatasets,
  neuralthread,
  CustApp,
  Math;

const
  csContextLen = 81;
  csTrainingFileName = 'datasets/tinystories.txt';
  csVocabSize  = 128; // Character based vocabulary/dictionary.
  csMinSampleSize = 3; // Minimum of 3 characters.

type

  { TTestFitLoading }

  TTestFitLoading = class(TCustomApplication)
  protected
    FDataset: TStringList;
    FDatasetSize: integer;
    FNN: THistoricalNets;
    NFit: TNeuralDataLoadingFit;
    FSampler: TNNetSamplerBase;
    FMaxPredictCharPos: integer;
    procedure LoadDataset;
    procedure DoRun; override;
  public
    procedure OnAfterEpoch(Sender: TObject);
    procedure OnAfterStep(Sender: TObject);
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetTestPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
  end;

  procedure TTestFitLoading.LoadDataset;
  // Streaming line-by-line loader. Replaces TStringList.LoadFromFile + two
  // backwards-traversal passes with a single forward pass: read a line,
  // apply min-length filter, lowercase + append sentinel, add to the list,
  // print progress every 100k lines. Constant working memory during load
  // (one line buffered at a time, plus the growing FDataset), one pass
  // instead of three, and visible progress so the user can see it's alive.
  var
    InputFile: TextFile;
    Line: string;
    LineNum: integer;
  begin
    WriteLn('Streaming dataset from ', csTrainingFileName, '...');
    Flush(Output);
    // Pre-size to avoid ~20 reallocation+copy cycles as the list grows.
    // 2.5M is an over-estimate for TinyStories (~2M stories); over-sizing
    // costs ~10MB of unused pointer slots, undersizing costs reallocations.
    FDataset.Capacity := 2500000;
    AssignFile(InputFile, csTrainingFileName);
    Reset(InputFile);
    try
      LineNum := 0;
      while not EOF(InputFile) do
      begin
        ReadLn(InputFile, Line);
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
      CloseFile(InputFile);
    end;
    FDatasetSize := FDataset.Count;
    WriteLn('Loaded dataset with ', FDatasetSize, ' rows');
    Flush(Output);
  end;

  procedure TTestFitLoading.DoRun;
  var
    W: TNNetLayer;
    I: integer;
  begin
    FDataset := TStringList.Create();
    LoadDataset();
    FNN := THistoricalNets.Create();
    NFit := TNeuralDataLoadingFit.Create();
    FMaxPredictCharPos := 81;
    FSampler := TNNetSamplerTopP.Create(0.4);
    FNN.AddLayer([
      TNNetInput.Create(csContextLen, 1, csVocabSize),
      TNNetAddPositionalEmbedding.Create(10000),
      TNNetConvolutionReLU.Create(32,1,0,1,0),
      TNNetConvolution.Create(128,13,0,13,0)
    ]);

    for I := 1 to 2 do
    begin
      FNN.AddTransformerBlockCAI(16, 512);
    end;

    FNN.AddLayer([
      TNNetFullConnectReLU.Create(128),
      TNNetFullConnectReLU.Create(csVocabSize),
      TNNetSoftMax.Create()
    ]);

    DebugThreadCount();
    FNN.DebugStructure;
    FNN.DebugWeights();

    WriteLn('Computing...');
    //NFit.MaxThreadNum := 1;
    NFit.LogEveryBatches := 1000;
    NFit.InitialLearningRate := 0.01;
    NFit.Inertia := 0;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.AvgWeightEpochCount := 1;
    NFit.OnAfterEpoch := @OnAfterEpoch;
    NFit.OnAfterStep := @OnAfterStep;
    NFit.FitLoading(
      FNN,
      {TrainingVolumesCount=}32000*3,
      {ValidationVolumesCount=}32000*3 div 20,
      {TestVolumesCount=}32000*3 div 20,
      {batchsize=}32,
      {epochs=}500,
      @GetTrainingPair, @GetValidationPair, @GetTestPair
    );
    FNN.DebugWeights();
    OnAfterEpoch(Self);
    FSampler.Free;
    NFit.Free;
    FNN.Free;
    FDataset.Free;
    Terminate;
  end;

  procedure TTestFitLoading.OnAfterEpoch(Sender: TObject);
  begin
    WriteLn('Testing.');
    WriteLn(GenerateStringFromChars(NFit.NN, 'once', FSampler),'.');
    WriteLn(GenerateStringFromChars(NFit.NN, 'lily loved ', FSampler),'.');
    WriteLn(GenerateStringFromChars(NFit.NN, 'she and he ', FSampler),'.');
    WriteLn(GenerateStringFromChars(NFit.NN, 'in the park ', FSampler),'.');
    WriteLn(GenerateStringFromChars(NFit.NN, 'billy ', FSampler),'.');
  end;

  procedure TTestFitLoading.OnAfterStep(Sender: TObject);
  begin
    //NFit.ThreadNN[0].DebugWeights();
  end;

  procedure TTestFitLoading.GetTrainingPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    SampleId: integer;
    SampleLen: integer;
    SampleCutPosition: integer;
    ExpectedTokenChar: char;
    ExpectedTokenInt: integer;
  begin
    // Make sure that expected input and output have the proper sizes.
    if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
    if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
    // Get the input sample
    SampleId := Random(FDatasetSize);
    SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
    SampleLen := Min(FMaxPredictCharPos, SampleLen);
    SampleCutPosition := Random(SampleLen-csMinSampleSize)+csMinSampleSize; // -1
    // The expected token is the next character in the string
    ExpectedTokenChar := FDataset[SampleId][SampleCutPosition+1];
    ExpectedTokenInt := Min(Ord(ExpectedTokenChar),pInput.Depth-1);
    // Encode the input and output volumes
    pInput.OneHotEncodingReversed(copy(FDataset[SampleId], 1, SampleCutPosition));
    pOutput.SetClassForSoftMax(ExpectedTokenInt);
    pOutput.Tag := ExpectedTokenInt;
  end;

  procedure TTestFitLoading.GetValidationPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    SampleId: integer;
    SampleLen: integer;
    SampleCutPosition: integer;
    ExpectedTokenChar: char;
    ExpectedTokenInt: integer;
  begin
    // Make sure that expected input and output have the proper sizes.
    if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
    if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
    // Get the input sample
    SampleId := Idx;
    SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
    SampleCutPosition := (Idx mod (1+SampleLen-csMinSampleSize))+csMinSampleSize-1;
    // The expected token is the next character in the string
    ExpectedTokenChar := FDataset[SampleId][SampleCutPosition+1];
    ExpectedTokenInt := Min(Ord(ExpectedTokenChar),pInput.Depth-1);
    // Encode the input and output volumes
    pInput.OneHotEncodingReversed(copy(FDataset[SampleId], 1, SampleCutPosition));
    pOutput.SetClassForSoftMax(ExpectedTokenInt);
    pOutput.Tag := ExpectedTokenInt;
  end;

  procedure TTestFitLoading.GetTestPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetValidationPair(Idx, ThreadId, pInput, pOutput);
  end;

var
  Application: TTestFitLoading;
begin
  Application := TTestFitLoading.Create(nil);
  Application.Title:='Nano Covolutional Based NLP Trained from File';
  Application.Run;
  Application.Free;
end.
