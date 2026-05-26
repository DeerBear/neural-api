program SimpleTransformer1MInfer;

{$APPTYPE CONSOLE}
(*
Inference-only driver for the KAN transformer.

Builds the same architecture as SimpleTransformer1M, loads weights from a
saved checkpoint via TNNet.LoadDataFromFile, then runs the
LockToInference + baseline-generate + CalibrateAlpha + recalibrated-
generate sequence -- skipping the training loop entirely.

Why this exists: training a 1.4M-param KAN transformer on TinyStories
takes ~2 days on the reference 4-thread non-AVX i5. Once an autosave is
available (autosave.nn / autosave_epoch26.nn / etc.), iterating on the
inference-time calibration -- or just regenerating with different
prompts -- shouldn't require sitting through training again.

Why this works: LoadDataFromFile populates weights into a pre-built
structure rather than recreating the structure from a serialised graph.
The freshly-built TKANNet here has the correct TNNetKANNormaliser layers
with their shared FBasis / FRNG / FInfo backrefs already wired up; only
the trainable parameters need to come from disk.

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
begin
  if ParamCount >= 1 then
    CheckpointFile := ParamStr(1)
  else
    CheckpointFile := csDefaultCheckpoint;

  WriteLn('Inference run using checkpoint: ', CheckpointFile);

  Dataset := TKANTransformerDataset.Create(csTrainingFileName, csContextLen);
  try
    // Dataset is loaded so CalibrateAlpha has something to score against;
    // pair-getter indices feed validation samples, same as training time.
    Dataset.LoadDataset;

    Net := BuildKANTransformer1M;
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
end.
