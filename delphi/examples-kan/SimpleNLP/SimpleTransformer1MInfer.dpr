program SimpleTransformer1MInfer;

{$APPTYPE CONSOLE}
(*
Inference-only driver for the KAN transformer.

Builds the same architecture as SimpleTransformer1M, loads weights from a
saved checkpoint via TNNet.LoadDataFromFile, then runs the
LockToInference + baseline-generate + fixed-alpha-generate sequence,
printing KAN telemetry (handover state) at each stage -- skipping the
training loop entirely. Per-pass SharpenAlpha calibration is disabled
(Option 3): alpha only steers the Phase-D NLMS target, not a single pass's
output, so it is set as a fixed hyperparameter and swept across runs.

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
  // Fixed inference-time SharpenAlpha (Option 3: per-pass calibration is
  // disabled). Sweep by changing this and re-running, or add more
  // Session.GenerateAtAlpha(...) calls below to compare alphas in one run.
  csInferenceAlpha = 1.1;

var
  Dataset: TKANTransformerDataset;
  Net: TKANNet;
  Session: TKANTransformerSession;
  CheckpointFile: string;
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
    // Dataset is loaded so RecommendedContextLen and CorpusHash are available
    // for the companion resolve / recover path below.
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
        Session.LockAndGenerate;
        Session.GenerateAtAlpha(csInferenceAlpha);
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
