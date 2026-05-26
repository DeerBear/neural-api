program SimpleTransformer1MTransplant;

{$APPTYPE CONSOLE}
(*
Q/K weight transplant experiment.

Loads two checkpoints:
  * Recent  (typically post-collapse): provides weights for most of
    the network -- FFN, output projection, V projections, the rest.
  * Healthy (pre-collapse): provides weights ONLY for the Q and K
    projections in the attention blocks, which is where the v1.0
    collapse manifested as anti-aligned degenerate projections.

The hypothesis the experiment tests: did the FFN and output projection
learn anything useful during the collapse epochs that would surface if
attention were healthy? If transplanted Q/K + post-collapse FFN
generates qualitatively better than either checkpoint alone, the
answer is yes -- the late-training updates weren't all wasted, only
the attention layers degraded. If transplanted output is no better
than the healthy checkpoint alone, the FFN's late-training updates
were either useless or actively conditioned on the broken attention
pattern (so the transplant breaks the FFN's input distribution).

Q/K layer indices for SimpleTransformer1M (2 KAN attention blocks):
  Block 1: Q = layer 4,   K = layer 5
  Block 2: Q = layer 181, K = layer 182

These are verified against the DebugStructure output: each block's
Q/K/V triplet consists of the first three TNNetPointwiseConvLinear
layers after the block's previous-output residual stream. V at index
6 / 183 is intentionally NOT transplanted -- the layer dump shows V
outputs varied normally even during collapse, so V weights are not
the broken part.

Usage:
  SimpleTransformer1MTransplant [recent.nn] [healthy.nn]

Defaults if no args are given:
  recent  = autosave.nn
  healthy = autosave_epoch26.nn
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
  kantransformersession in 'kantransformersession.pas',
  kanweightremix in 'kanweightremix.pas';

const
  csTrainingFileName  = 'datasets/tinystories.txt';
  csDefaultRecent     = 'autosave.nn';
  csDefaultHealthy    = 'autosave_epoch26.nn';
  // Q/K layer indices for the SimpleTransformer1M architecture.
  // Verified against the DebugStructure dump produced after
  // BuildKANTransformer1M. If kantransformerarch changes (extra
  // pre-attention layers, different block count), these must be
  // updated. The transplant DPR is intentionally tied to one known
  // architecture; the kanweightremix helper itself is generic.
  csQKLayers: array[0..3] of integer = (4, 5, 181, 182);

var
  Dataset: TKANTransformerDataset;
  Net: TKANNet;
  Session: TKANTransformerSession;
  RecentFile, HealthyFile: string;
  ValidationCount: integer;
begin
  if ParamCount >= 2 then
  begin
    RecentFile  := ParamStr(1);
    HealthyFile := ParamStr(2);
  end
  else
  begin
    RecentFile  := csDefaultRecent;
    HealthyFile := csDefaultHealthy;
  end;

  WriteLn('Q/K transplant run');
  WriteLn('  Recent checkpoint:  ', RecentFile);
  WriteLn('  Healthy checkpoint: ', HealthyFile);
  WriteLn(Format('  Q/K layers receiving healthy weights: [%d, %d, %d, %d]',
    [csQKLayers[0], csQKLayers[1], csQKLayers[2], csQKLayers[3]]));

  if not FileExists(RecentFile) then
  begin
    WriteLn('ERROR: recent checkpoint not found: ', RecentFile);
    Halt(1);
  end;
  if not FileExists(HealthyFile) then
  begin
    WriteLn('ERROR: healthy checkpoint not found: ', HealthyFile);
    Halt(1);
  end;

  Dataset := TKANTransformerDataset.Create(csTrainingFileName, csContextLen);
  try
    Dataset.LoadDataset;

    // Transplant must build at the checkpoint's training-time ContextLen.
    // The Q/K indices below (4, 5, 181, 182) are also tied to that
    // structure: changing ContextLen would not shift them (those layer
    // positions are determined by attention head count, not context size),
    // but the loaded weight shapes would mismatch on positional/input
    // layers. Keep this aligned with the training run that produced the
    // input checkpoints.
    Net := BuildKANTransformer1M(csContextLen);
    try
      Dataset.BindNetwork(Net);

      DebugThreadCount();
      Net.DebugStructure;

      WriteLn('Loading recent weights from ', RecentFile, '...');
      Net.LoadDataFromFile(RecentFile);

      WriteLn('Transplanting Q/K weights from ', HealthyFile, '...');
      LoadLayersFromFile(Net, HealthyFile, csQKLayers);

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
