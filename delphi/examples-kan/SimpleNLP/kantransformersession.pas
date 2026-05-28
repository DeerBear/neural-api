unit kantransformersession;

(*
Training + inference session for the KAN transformer example.

Encapsulates everything that wraps the network and dataset: the
NeuralDataLoadingFit instance, the per-step weight clipping stabiliser,
the per-epoch plateau-based early stop, the generation sampler, and the
LockToInference → CalibrateAlpha → re-generate sequence.

Two entry points:
  * Train() — full training loop with stabilisers and plateau detection.
  * LockAndGenerate() / CalibrateAndGenerate() — inference-time
    operations. Callable without prior Train() when a saved checkpoint
    has been loaded via FNN.LoadDataFromFile into a freshly-built
    architecture.

Stays out of the dataset getters' inner loops: pair-getters live on
TKANTransformerDataset; this unit only references them via method
pointers passed into FitLoading / CalibrateAlpha.
*)

interface

uses
  Classes, SysUtils, Math,
  neuralvolume, neuralnetwork, neuralfit, neuralthread, neuraldatasets,
  neuralkantypes, neuralkanattention,
  kantransformerdata;

const
  // Energy-conserving per-neuron weight clipping threshold. Per-neuron
  // weights with |w| > csWeightClipMax get clipped to ±csWeightClipMax
  // and the excess magnitude is redistributed across the remaining
  // weights in the same neuron, proportional to their existing
  // magnitude. Preserves the L1 norm of each neuron's weight vector
  // while preventing any single weight from initiating runaway
  // amplification.
  //
  // Threshold rationale: observed Q/K projection max weights at epoch 28
  // (v1.0 collapse) were ~0.30; at epoch 16 (healthy) they were ~0.15.
  // 0.20 keeps weights well below the runaway threshold while leaving
  // headroom for legitimate learning. Tune for v1.1.
  csWeightClipMax: TNeuralFloat = 0.20;

  // Looser threshold for Q/K/V projection layers in attention blocks.
  // Attention dot products scale as Q*K, so capping Q/K at the same
  // magnitude as the rest of the network (0.20) keeps softmax outputs
  // near uniform 1/H = degenerate attention. The previous v1.1 run hit
  // exactly this failure: Q/K weights settled around 0.11, dot products
  // landed at 0.03-0.10, softmax outputs at 0.164-0.169 (uniform = 0.167).
  // 0.40 gives the projections room to grow into a regime where dot
  // products are meaningful (target: 0.5-2.0 range after MulByConstant).
  csQKWeightClipMax: TNeuralFloat = 0.40;

type
  TKANTransformerSession = class
  private
    FNN: TKANNet;
    FDataset: TKANTransformerDataset;
    FNFit: TNeuralDataLoadingFit;
    FSampler: TNNetSamplerBase;
    // Plateau-based early stopping: track best ValidationLoss and the
    // epoch it was achieved. If FPlateauWindow epochs pass without a
    // new record, the run is treated as converged and ShouldQuit is
    // signalled.
    FBestLoss: TNeuralFloat;
    FBestLossEpoch: integer;
    FPlateauWindow: integer;
    procedure OnAfterEpoch(Sender: TObject);
    procedure OnAfterStep(Sender: TObject);
    procedure GenerateSamples;
  public
    constructor Create(ANN: TKANNet; ADataset: TKANTransformerDataset);
    destructor Destroy; override;
    procedure Train(TrainingCount, ValidationCount, TestCount,
      BatchSize, Epochs: integer);
    // Engage the KAN attention path on the trained (or weight-loaded)
    // network and generate the canonical prompt set with baseline alpha.
    procedure LockAndGenerate;
    // Run continuous EMA-driven calibration of SharpenAlpha against
    // validation cross-entropy, then re-generate. Must be called after
    // LockAndGenerate (or after FNN is otherwise in inference mode).
    procedure CalibrateAndGenerate(ValidationCount: integer);
    property PlateauWindow: integer
      read FPlateauWindow write FPlateauWindow;
  end;

implementation

// Energy-conserving weight clipping. When a weight exceeds MaxAbs in
// absolute value, clip it to ±MaxAbs and redistribute the excess
// magnitude across the remaining weights in the same volume,
// proportional to their current magnitude. Preserves the L1 norm
// (sum of |w_i|) while bounding max(|w_i|) <= MaxAbs.
//
// The Mechanism-#1 design philosophy applied to network weights:
// when a single channel tries to amplify, the optimizer can't drop
// the excess into the void -- it has to spread it across other
// channels, which disrupts the positive-feedback runaway that broke
// the v1.0 training at epoch 28.
//
// Single-pass: after redistribution some weights may slightly exceed
// MaxAbs (bounded by their proportional share). Iterate-to-convergence
// is a v1.1.1 refinement; this is the minimal correct implementation.
procedure ClipAndSpreadWeights(W: TNNetVolume; const MaxAbs: TNeuralFloat);
var
  I: integer;
  Weight, AbsWeight, Excess, TotalExcess, UnclippedL1, ScaleFactor: TNeuralFloat;
  IsClipped: array of boolean;
begin
  if (W = nil) or (W.Size = 0) or (MaxAbs <= 0) then exit;

  SetLength(IsClipped, W.Size);
  TotalExcess := 0;

  for I := 0 to W.Size - 1 do
  begin
    Weight := W.FData[I];
    AbsWeight := Abs(Weight);
    if AbsWeight > MaxAbs then
    begin
      Excess := AbsWeight - MaxAbs;
      TotalExcess := TotalExcess + Excess;
      if Weight > 0 then W.FData[I] := MaxAbs
      else W.FData[I] := -MaxAbs;
      IsClipped[I] := True;
    end
    else
      IsClipped[I] := False;
  end;

  if TotalExcess = 0 then exit;

  UnclippedL1 := 0;
  for I := 0 to W.Size - 1 do
    if not IsClipped[I] then
      UnclippedL1 := UnclippedL1 + Abs(W.FData[I]);

  // Degenerate cases: all weights clipped, or unclipped weights are zero.
  // Leave clipped values in place; nowhere to spread the excess.
  if UnclippedL1 <= 0 then exit;

  ScaleFactor := TotalExcess / UnclippedL1;
  for I := 0 to W.Size - 1 do
    if not IsClipped[I] then
    begin
      Weight := W.FData[I];
      if Weight > 0 then
        W.FData[I] := Weight + Abs(Weight) * ScaleFactor
      else if Weight < 0 then
        W.FData[I] := Weight - Abs(Weight) * ScaleFactor;
    end;
end;

constructor TKANTransformerSession.Create(ANN: TKANNet;
  ADataset: TKANTransformerDataset);
begin
  inherited Create;
  FNN := ANN;
  FDataset := ADataset;
  FNFit := TNeuralDataLoadingFit.Create;
  FSampler := TNNetSamplerTopP.Create(0.4);
  FBestLoss := 1e30;
  FBestLossEpoch := 0;
  FPlateauWindow := 10;
end;

destructor TKANTransformerSession.Destroy;
begin
  FSampler.Free;
  FNFit.Free;
  inherited Destroy;
end;

procedure TKANTransformerSession.Train(TrainingCount, ValidationCount,
  TestCount, BatchSize, Epochs: integer);
begin
  FNFit.LogEveryBatches := 100;
  FNFit.InitialLearningRate := 0.01;
  FNFit.Inertia := 0;
  FNFit.LearningRateDecay := 0;
  FNFit.L2Decay := 0;
  FNFit.EnableClassComparison();
  FNFit.EnableDefaultLoss();
  FNFit.AvgWeightEpochCount := 1;
  FNFit.OnAfterEpoch := OnAfterEpoch;
  FNFit.OnAfterStep := OnAfterStep;
  FNFit.FitLoading(
    FNN,
    TrainingCount,
    ValidationCount,
    TestCount,
    BatchSize,
    Epochs,
    FDataset.GetTrainingPair,
    FDataset.GetValidationPair,
    FDataset.GetTestPair
  );
  FNN.DebugWeights;
end;

procedure TKANTransformerSession.LockAndGenerate;
var
  Snapshot: TKANCoeffSnapshot;
begin
  // After this call, every TNNetKANNormaliser in FNN switches from
  // passthrough to its B-spline normaliser. LockToInference is one-way --
  // no further training is permitted on FNN.
  FNN.LockToInference;
  WriteLn('--- KAN attention engaged; post-training generation (baseline alpha=1.1): ---');
  // Inference-mode Compute mutates FHead.Coeffs on every forward pass
  // (NLMS Phase M/D, see TNNetKANNormaliser.Compute step 7). Snapshot
  // around generation so the loaded checkpoint is left untouched after
  // this call -- LockAndGenerate is a pure read of the saved weights.
  Snapshot := FNN.SnapshotAllCoeffs;
  try
    GenerateSamples;
  finally
    FNN.RestoreAllCoeffs(Snapshot);
  end;
end;

procedure TKANTransformerSession.CalibrateAndGenerate(ValidationCount: integer);
var
  Snapshot: TKANCoeffSnapshot;
begin
  WriteLn('--- Calibrating SharpenAlpha against validation loss ---');
  // CalibrateAlpha defaults to PreserveCoeffs=true + UseBestAlpha=true:
  // only the SharpenAlpha hyperparameter persists from the sweep, and
  // the alpha selected is the one with the lowest observed validation
  // loss across iterations.
  FNN.CalibrateAlpha(FDataset.GetValidationPair, ValidationCount);
  WriteLn('--- Post-calibration generation: ---');
  // Same non-destructive pattern as LockAndGenerate: the post-cal
  // generation samples must not drift coefficients either, otherwise a
  // subsequent inspection (or a re-run of CalibrateAndGenerate) would
  // see a network that differs from the calibrated state.
  Snapshot := FNN.SnapshotAllCoeffs;
  try
    GenerateSamples;
  finally
    FNN.RestoreAllCoeffs(Snapshot);
  end;
end;

procedure TKANTransformerSession.GenerateSamples;
begin
  WriteLn('Testing.');
  WriteLn(GenerateStringFromChars(FNN, 'once', FSampler), '.');
  WriteLn(GenerateStringFromChars(FNN, 'lily loved ', FSampler), '.');
  WriteLn(GenerateStringFromChars(FNN, 'she and he ', FSampler), '.');
  WriteLn(GenerateStringFromChars(FNN, 'in the park ', FSampler), '.');
  WriteLn(GenerateStringFromChars(FNN, 'billy ', FSampler), '.');
end;

procedure TKANTransformerSession.OnAfterEpoch(Sender: TObject);
begin
  GenerateSamples;

  // Plateau check. Only meaningful inside the training loop (FNFit
  // populates ValidationLoss / CurrentEpoch); skip otherwise.
  if Sender = FNFit then
  begin
    if FNFit.ValidationLoss < FBestLoss then
    begin
      FBestLoss := FNFit.ValidationLoss;
      FBestLossEpoch := FNFit.CurrentEpoch;
    end
    else if (FNFit.CurrentEpoch - FBestLossEpoch) >= FPlateauWindow then
    begin
      WriteLn(Format(
        'Plateau: no ValidationLoss improvement for %d epochs (best %.4f at epoch %d). Stopping.',
        [FPlateauWindow, FBestLoss, FBestLossEpoch]));
      FNFit.ShouldQuit := true;
    end;
  end;
end;

procedure TKANTransformerSession.OnAfterStep(Sender: TObject);
var
  LayerIdx, NeuronIdx, AttIdx: integer;
  Layer: TNNetLayer;
  Info: TKANAttentionLayerInfo;
  QKVLayers: TList;
  ClipMax: TNeuralFloat;
begin
  // Apply energy-conserving weight clipping per-neuron across the entire
  // network after each batch. Per-neuron (rather than per-layer) preserves
  // the relative weight budgets between output channels; each channel's
  // weights spread among themselves but channels stay independent.
  //
  // Two thresholds: csQKWeightClipMax for Q/K/V projection layers in
  // attention blocks (those need more dynamic range to produce non-trivial
  // dot products), csWeightClipMax for everything else.
  QKVLayers := TList.Create;
  try
    for AttIdx := 0 to FNN.AttentionLayers.Count - 1 do
    begin
      Info := TKANAttentionLayerInfo(FNN.AttentionLayers[AttIdx]);
      if Info.QProjection <> nil then QKVLayers.Add(Info.QProjection);
      if Info.KProjection <> nil then QKVLayers.Add(Info.KProjection);
      if Info.VProjection <> nil then QKVLayers.Add(Info.VProjection);
    end;

    for LayerIdx := 0 to FNN.CountLayers - 1 do
    begin
      Layer := FNN.Layers[LayerIdx];
      if Layer.Neurons.Count = 0 then continue;
      if QKVLayers.IndexOf(Layer) >= 0
        then ClipMax := csQKWeightClipMax
        else ClipMax := csWeightClipMax;
      for NeuronIdx := 0 to Layer.Neurons.Count - 1 do
        ClipAndSpreadWeights(Layer.Neurons[NeuronIdx].Weights, ClipMax);
    end;
  finally
    QKVLayers.Free;
  end;
end;

end.
