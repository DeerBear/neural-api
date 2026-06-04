program SimpleTransformer1MInfer;

{$APPTYPE CONSOLE}
(*
Inference-only driver for the KAN transformer.

Builds the same architecture as SimpleTransformer1M, loads weights from a
saved checkpoint via TNNet.LoadDataFromFile, then runs the
LockToInference + baseline-generate + CalibrateAlpha + recalibrated-
generate sequence -- skipping the training loop entirely.

Context length is resolved from the checkpoint's companion file (see
checkpointcompanion):
  * companion present AND corpus + nn MD5s both match -> use its context;
  * companion absent OR either MD5 mismatched           -> recover by
    recomputing the context from the corpus, narrating the decision.
This keeps inference building at the SAME context the checkpoint was trained
at (positional embeddings are sized to it), and tracks the trainer when a
companion is available, while still loading older companion-less checkpoints.

Why this exists: training a 1.4M-param KAN transformer on TinyStories takes a
long time. Once an autosave is available, iterating on inference-time
calibration -- or just regenerating -- shouldn't require retraining.

Usage:
  SimpleTransformer1MInfer [checkpoint.nn]

Defaults to 'autosave.nn' if no argument is given.
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
  mmaptextdataset in 'mmaptextdataset.pas',
  prefetchloader in 'prefetchloader.pas',
  checkpointcompanion in 'checkpointcompanion.pas',
  kantransformersession in 'kantransformersession.pas';

const
  csTrainingFileName = 'datasets/tinystories.txt';
  csDefaultCheckpoint = 'autosave.nn';

var
  Dataset: TKANTransformerDataset;
  Net: TKANNet;
  Session: TKANTransformerSession;
  CheckpointFile: string;
  ValidationCount: integer;
  Comp: TCheckpointCompanion;
  ContextLen: integer;
  Recovered: boolean;
begin
  if ParamCount >= 1 then
    CheckpointFile := ParamStr(1)
  else
    CheckpointFile := csDefaultCheckpoint;

  WriteLn('Inference run using checkpoint: ', CheckpointFile);

  Dataset := TKANTransformerDataset.Create(csTrainingFileName, csContextLen);
  try
    // Dataset is loaded so CalibrateAlpha has something to score against, and
    // so RecommendedContextLen is available for the recover path below.
    Dataset.LoadDataset;

    // --- Resolve context length via the shared companion mechanism ---
    // Same class the trainer writes with, so the two cannot drift. Present and
    // content-consistent (corpus hash + nn MD5) -> stored context; otherwise
    // recover to the recomputed context. Resolve returns the line to print.
    Comp := TCheckpointCompanion.Create(CheckpointFile);
    try
      WriteLn('  ' + Comp.Resolve(Dataset.CorpusHash,
        Dataset.RecommendedContextLen, ContextLen, Recovered));
    finally
      Comp.Free;
    end;

    // Build at the resolved context. If a recovered value disagrees with a
    // stale checkpoint's real context, the load below fails on the
    // positional-embedding shape -- the intended loud staleness signal.
    Net := BuildKANTransformer1M(ContextLen);
    try
      Dataset.BindNetwork(Net);

      DebugThreadCount();
      Net.DebugStructure;

      WriteLn('Loading weights from ', CheckpointFile, '...');
      if not FileExists(CheckpointFile) then
      begin
        WriteLn('ERROR: checkpoint file not found: ', CheckpointFile);
        Halt(1);
      end;
      Net.LoadDataFromFile(CheckpointFile);
      Net.DebugWeights;

      Session := TKANTransformerSession.Create(Net, Dataset);
      try
        ValidationCount := 32000 * 3 div 20;
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
  WriteLn('Press ENTER to exit.');
  ReadLn;
end.
