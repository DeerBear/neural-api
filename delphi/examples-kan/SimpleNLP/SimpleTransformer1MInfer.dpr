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
  kanmmapdataset in 'kanmmapdataset.pas',
  kanprefetch in 'kanprefetch.pas',
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
  StratifiedSampleCount: integer;
begin
  if ParamCount >= 1 then
    CheckpointFile := ParamStr(1)
  else
    CheckpointFile := csDefaultCheckpoint;

  // Optional 2nd arg: teacher-forced samples for the stratified NLL read.
  // Each is one forward pass and the read runs twice (baseline + KAN), so
  // keep it bounded on CPU. Default 16000 -- enough to populate the rare
  // bucket; pass a smaller number for a quicker, noisier read.
  if ParamCount >= 2 then
    StratifiedSampleCount := StrToIntDef(ParamStr(2), 16000)
  else
    StratifiedSampleCount := 16000;

  WriteLn('Inference run using checkpoint: ', CheckpointFile);

  Dataset := TKANTransformerDataset.Create(csTrainingFileName, csContextLen);
  try
    // Dataset is loaded so CalibrateAlpha has something to score against;
    // pair-getter indices feed validation samples, same as training time.
    Dataset.LoadDataset;

    // Inference must build the network with the SAME context length the
    // checkpoint was trained at; positional-embedding weights are sized to
    // ContextLen, so a mismatch fails LoadDataFromFile. The training driver
    // (SimpleTransformer1M) builds from Dataset.RecommendedContextLen (84 on
    // the current corpus), so match that here. The legacy csContextLen (81)
    // is only correct for old checkpoints up to autosave_epoch26.nn.
    Net := BuildKANTransformer1M(Dataset.RecommendedContextLen);
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

        // Rare-bucket stratified NLL: baseline vs KAN. The network trains as
        // vanilla softmax attention; the KAN normaliser only activates after
        // LockToInference. So measure the teacher-forced, frequency-stratified
        // NLL BEFORE locking (softmax baseline) and AFTER the inference warm-up
        // passes (KAN-mode). The KAN hypothesis is the per-bucket delta between
        // the two -- especially on the rare third.
        WriteLn('');
        WriteLn('=== Stratified NLL: BASELINE (softmax, pre-LockToInference) ===');
        Session.EvaluateStratifiedNLL(StratifiedSampleCount);

        Session.LockAndGenerate;
        Session.CalibrateAndGenerate(ValidationCount);

        // Post-lock + post-generation: the KAN has now had forward passes to
        // progress past Phase M (softmax mimicry) toward takeover / Phase D.
        // If this reads ~identical to baseline, either the KAN is still
        // mimicking (needs more warm-up) or the inference path isn't taking
        // over -- both worth knowing before trusting the delta.
        WriteLn('');
        WriteLn('=== Stratified NLL: KAN (post-LockToInference) ===');
        Session.EvaluateStratifiedNLL(StratifiedSampleCount);
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
