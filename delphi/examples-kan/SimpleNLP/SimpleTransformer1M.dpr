program SimpleTransformer1M;

{$APPTYPE CONSOLE}
(*
KAN-attention variant of the SimpleTransformer NLP example, scaled to
~1.4M parameters so the architecture has enough capacity for a serious
KAN-vs-softmax comparison.

Identical to the base SimpleTransformer except:
  * THistoricalNets -> TKANNet  (the KAN-aware network class).
  * Embedding dim raised from 128 to 256 by changing the seq-mixing
    TNNetConvolution's output channels. 256 / 16 heads = 16 dims per
    head; divides cleanly. Total params ~1.39M, almost all in the two
    transformer blocks (~525K each).
  * The transformer block is built by hand instead of via
    AddTransformerBlockCAI, so that the per-head softmax inside the
    attention sub-chain can be replaced by TNNetKANNormaliser via
    TKANNet.AddKANSelfAttention. The rest of the block (residual sum,
    FFN, second residual, activations) mirrors the layout of
    TNNet.AddTransformerBlockCAI byte-for-byte.
  * FNN.LockToInference is called after FitLoading completes, so the
    one post-training GenerateStringFromChars run engages the KAN
    attention path; everything before that (training + per-epoch
    sampling) runs through the bit-identical softmax fallback.

Paired with delphi/examples/SimpleNLP/SimpleTransformer1M.dpr (same
architecture, softmax-attention only) for a single-architectural-
variable A/B. KAN's added spline coefficients are ~2K total
(KnotCount 64 x Heads 16 x AttentionLayers 2) -- about 0.15% of the
~1.4M total, so the two variants are essentially param-matched.

Runs only once the KAN implementation stubs have all been filled --
in v1 several TNNetKANNormaliser pipeline methods (NLMSPhaseM/D,
the Compute orchestrator, the AddKANSelfAttention builder body) are
still stubs that raise EKANBadState.

Copyright (C) 2023 Joao Paulo Schwarz Schuler
*)


uses
  Classes,
  SysUtils,
  neuralnetwork in '..\..\neural\neuralnetwork.pas',
  neuralvolume in '..\..\neural\neuralvolume.pas',
  neuralfit in '..\..\neural\neuralfit.pas',
  neuraldatasets in '..\..\neural\neuraldatasets.pas',
  neuralthread in '..\..\neural\neuralthread.pas',
  CustApp in '..\..\examples\CustApp.pas',
  Math,
  neuralab in '..\..\neural\neuralab.pas',
  neuralabfun in '..\..\neural\neuralabfun.pas',
  neuralbit in '..\..\neural\neuralbit.pas',
  neuralbyteprediction in '..\..\neural\neuralbyteprediction.pas',
  neuralcache in '..\..\neural\neuralcache.pas',
  neuralgeneric in '..\..\neural\neuralgeneric.pas',
  neuralsimd in '..\..\neural\neuralsimd.pas',
  neuralkantypes in '..\..\neural\neuralkantypes.pas',
  neuralkanbasis in '..\..\neural\neuralkanbasis.pas',
  neuralkannormaliser in '..\..\neural\neuralkannormaliser.pas',
  neuralkanattention in '..\..\neural\neuralkanattention.pas';

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
    FNN: TKANNet;
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
  var
    RowCnt: integer;
  begin
    FDataset.LoadFromFile(csTrainingFileName);
    FDatasetSize := FDataset.Count;
    for RowCnt := FDatasetSize-1 downto 0 do
    begin
      // removes too short strings
      if Length(FDataset[RowCnt])<csMinSampleSize then FDataset.Delete(RowCnt);
    end;
    FDatasetSize := FDataset.Count;
    for RowCnt := FDatasetSize-1 downto 0 do
    begin
      // removes too short strings
      FDataset[RowCnt] := LowerCase(FDataset[RowCnt]) + chr(1);
    end;
    WriteLn('Loaded dataset with ', FDatasetSize, ' rows');
  end;

  procedure TTestFitLoading.DoRun;
  var
    W: TNNetLayer;
    I: integer;
    PrevLayer, Attended, AttendedPlusPrev: TNNetLayer;
    EmbeddingDim: integer;
  begin
    FDataset := TStringList.Create();
    LoadDataset();
    FNN := TKANNet.Create();
    NFit := TNeuralDataLoadingFit.Create();
    FMaxPredictCharPos := 81;
    FSampler := TNNetSamplerTopP.Create(0.4);
    FNN.AddLayer([
      TNNetInput.Create(csContextLen, 1, csVocabSize),
      TNNetAddPositionalEmbedding.Create(10000),
      TNNetConvolutionReLU.Create(32,1,0,1,0),
      TNNetConvolution.Create(256,13,0,13,0)
    ]);

    // Two KAN-attention transformer blocks. Layout mirrors
    // TNNet.AddTransformerBlockCAI(Heads=16, IntermediateDim=512, pActFn=
    // TNNetSignedSquareRoot1) but swaps the per-head softmax in the
    // attention sub-chain for TNNetKANNormaliser via AddKANSelfAttention.
    for I := 1 to 2 do
    begin
      PrevLayer := FNN.GetLastLayer();
      EmbeddingDim := PrevLayer.Output.Depth;
      Attended := FNN.AddKANSelfAttention({InitialHeads=}16);
      AttendedPlusPrev := FNN.AddLayer( TNNetSum.Create([Attended, PrevLayer]) );
      AttendedPlusPrev := FNN.AddLayer( TNNetSignedSquareRoot1.Create() );
      FNN.AddLayer( TNNetPointwiseConvReLU.Create(512, 1) );
      FNN.AddLayer( TNNetSignedSquareRoot1.Create() );
      FNN.AddLayer( TNNetPointwiseConvLinear.Create(EmbeddingDim, 1) );
      FNN.AddLayer( TNNetSignedSquareRoot1.Create() );
      FNN.AddLayer( TNNetSum.Create([FNN.GetLastLayer(), AttendedPlusPrev]) );
      FNN.AddLayer( TNNetSignedSquareRoot1.Create() );
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
    NFit.LogEveryBatches := 100;     // ~30 log lines/epoch instead of ~3
    NFit.InitialLearningRate := 0.01;
    NFit.Inertia := 0;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.AvgWeightEpochCount := 1;
    NFit.OnAfterEpoch := OnAfterEpoch;
    NFit.OnAfterStep := OnAfterStep;
    NFit.FitLoading(
      FNN,
      {TrainingVolumesCount=}32000*3,
      {ValidationVolumesCount=}32000*3 div 20,
      {TestVolumesCount=}32000*3 div 20,
      {batchsize=}32,
      {epochs=}500,
      GetTrainingPair, GetValidationPair, GetTestPair
    );
    FNN.DebugWeights();

    // Training is complete: engage the KAN attention path. After this
    // call, every TNNetKANNormaliser in FNN switches from passthrough to
    // its B-spline normaliser. The final OnAfterEpoch below therefore
    // generates with KAN attention active. LockToInference is one-way --
    // no further training is permitted on FNN.
    FNN.LockToInference;
    WriteLn('--- KAN attention engaged; post-training generation: ---');

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
  Application.Title:='SimpleTransformer1M with KAN Attention (TinyStories)';
  Application.Run;
  Application.Free;
end.
