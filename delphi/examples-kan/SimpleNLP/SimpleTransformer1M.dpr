program SimpleTransformer1M;

{$APPTYPE CONSOLE}
(*
KAN-attention variant of the SimpleTransformer NLP example, scaled to
~1.4M parameters so the architecture has enough capacity for a serious
KAN-vs-softmax comparison.

This program is now a thin orchestrator. The actual machinery is split
across three sibling units so the inference-only program can share
everything except the training call:

  * kantransformerarch     -- architecture construction (BuildKANTransformer1M).
  * kantransformerdata     -- TinyStories streaming loader and pair-getters.
  * kantransformersession  -- training loop, plateau early-stop, weight
                              clipping stabiliser, LockToInference +
                              CalibrateAlpha sequence, generation sampler.

Identical to the base SimpleTransformer except:
  * THistoricalNets -> TKANNet  (the KAN-aware network class).
  * Embedding dim raised from 128 to 256 by changing the seq-mixing
    TNNetConvolution's output channels. 256 / 16 heads = 16 dims per
    head; divides cleanly. Total params ~1.39M, almost all in the two
    transformer blocks (~525K each).
  * The transformer block is built by hand instead of via
    AddTransformerBlockCAI, so that the per-head softmax inside the
    attention sub-chain can be replaced by TNNetKANNormaliser via
    TKANNet.AddKANSelfAttention.
  * FNN.LockToInference is called after FitLoading completes, so the
    post-training generation engages the KAN attention path; everything
    before that (training + per-epoch sampling) runs through the bit-
    identical softmax fallback.

Paired with delphi/examples/SimpleNLP/SimpleTransformer.lpr (the softmax-
attention baseline) for a single-architectural-variable A/B comparison.

Copyright (C) 2023 Joao Paulo Schwarz Schuler
*)

uses
  Classes,
  SysUtils,
  Math,
  neuralnetwork in '..\..\neural\neuralnetwork.pas',
  neuralvolume in '..\..\neural\neuralvolume.pas',
  neuralfit in '..\..\neural\neuralfit.pas',
  neuraldatasets in '..\..\neural\neuraldatasets.pas',
  neuralthread in '..\..\neural\neuralthread.pas',
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
  neuralkanattention in '..\..\neural\neuralkanattention.pas',
  kantransformerarch in 'kantransformerarch.pas',
  kantransformerdata in 'kantransformerdata.pas',
  kantransformersession in 'kantransformersession.pas';

const
  csTrainingFileName = 'datasets/tinystories.txt';

var
  Dataset: TKANTransformerDataset;
  Net: TKANNet;
  Session: TKANTransformerSession;
  ValidationCount: integer;
begin
  Dataset := TKANTransformerDataset.Create(csTrainingFileName, csContextLen);
  try
    Dataset.LoadDataset;

    // Use the data-driven context length derived from the log-log mean
    // of line lengths. Robust to heavy-tailed length distributions;
    // economists use the same transform when comparing wildly different
    // unit scales (GDP vs population, etc.).
    Net := BuildKANTransformer1M(Dataset.RecommendedContextLen);
    try
      Dataset.BindNetwork(Net);

      DebugThreadCount();
      Net.DebugStructure;
      Net.DebugWeights();

      WriteLn('Computing...');
      Session := TKANTransformerSession.Create(Net, Dataset);
      try
        ValidationCount := 32000 * 3 div 20;
        Session.Train(
          {TrainingCount=}     32000 * 3,
          {ValidationCount=}   ValidationCount,
          {TestCount=}         32000 * 3 div 20,
          {BatchSize=}         32,
          {Epochs=}            500
        );
        Session.LockAndGenerate;
        Session.CalibrateAndGenerate(ValidationCount);
      finally
        Session.Free;
      end;
    finally
      Net.Free;
    end;
  finally
    Dataset.Free;
  end;
end.
