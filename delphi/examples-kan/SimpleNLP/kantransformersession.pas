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

// Energy-conserving weight clipping (L2). When a weight exceeds MaxAbs
// in absolute value, clip it to ±MaxAbs and redistribute the excess
// L2-energy (W_i^2 - MaxAbs^2) across the remaining weights in the same
// volume, proportional to each weight's share of the unclipped subset's
// energy (W_j^2 / Sum(W_k^2)). Preserves Sum(W_i^2) (the L2 energy of
// the neuron) while bounding max(|w_i|) <= MaxAbs.
//
// The proportional-to-energy distribution is a uniform multiplicative
// rescale of the unclipped subset:
//   W_j := W_j * sqrt(1 + TotalExcessEnergy / UnclippedEnergy)
// This preserves the L2 shape of the unclipped weights exactly --
// relative ratios are unchanged, only the overall scale grows to
// absorb the excess.
//
// The Mechanism-#1 design philosophy applied to network weights:
// when a single channel tries to amplify, the optimizer can't drop
// the excess into the void -- it has to spread it across other
// channels, which disrupts the positive-feedback runaway that broke
// the v1.0 training at epoch 28.
//
// Degenerate-neuron handling: if the unclipped subset has zero energy
// (e.g. all weights at the cap, or the rest are exactly zero), the
// excess is dropped rather than redistributed. Conserving energy into
// an already-collapsed neuron would just feed the runaway.
//
// Single-pass: after rescale some unclipped weights may exceed MaxAbs
// (bounded by their proportional share of the bump). Iterate-to-
// convergence is a v1.1.1 refinement; this is the minimal correct form.
procedure ClipAndSpreadWeights(W: TNNetVolume; const MaxAbs: TNeuralFloat);
var
  I: integer;
  Weight, AbsWeight, TotalExcessEnergy, UnclippedEnergy, EnergyFactor: TNeuralFloat;
  IsClipped: array of boolean;
begin
  if (W = nil) or (W.Size = 0) or (MaxAbs <= 0) then exit;

  SetLength(IsClipped, W.Size);
  TotalExcessEnergy := 0;

  for I := 0 to W.Size - 1 do
  begin
    Weight := W.FData[I];
    AbsWeight := Abs(Weight);
    if AbsWeight > MaxAbs then
    begin
      // L2-energy of the clipped weight beyond the cap.
      TotalExcessEnergy := TotalExcessEnergy + (AbsWeight*AbsWeight - MaxAbs*MaxAbs);
      if Weight > 0 then W.FData[I] := MaxAbs
      else W.FData[I] := -MaxAbs;
      IsClipped[I] := True;
    end
    else
      IsClipped[I] := False;
  end;

  if TotalExcessEnergy <= 0 then exit;

  UnclippedEnergy := 0;
  for I := 0 to W.Size - 1 do
    if not IsClipped[I] then
      UnclippedEnergy := UnclippedEnergy + W.FData[I]*W.FData[I];

  // Degenerate case: nowhere to spread the excess. Drop it.
  if UnclippedEnergy <= 0 then exit;

  EnergyFactor := Sqrt(1 + TotalExcessEnergy / UnclippedEnergy);
  for I := 0 to W.Size - 1 do
    if not IsClipped[I] then
      W.FData[I] := W.FData[I] * EnergyFactor;
end;

// Diagnostic: dump max|W|, std|W|, and "% of weights pinned at the cap"
// for one layer. Computed across every weight in every neuron of the
// layer -- one line of output per call. The three numbers together
// distinguish:
//   * Healthy:        max < cap, std non-trivial, pinned% ~ 0
//   * Stuck small:    max << cap, std small, pinned% = 0
//   * Saturating:     max == cap, std non-trivial, pinned% low but rising
//   * Homogenized:    max == cap, std collapsing, pinned% climbing -> 100
// The collapse signature we're hunting is "homogenized" -- specifically
// std collapsing while pinned% rises, indicating weights migrating into
// the clipped subset and being held there by the clipper.
procedure DumpLayerWeightStats(Layer: TNNetLayer; const LayerName: string;
  const ClipMax: TNeuralFloat);
var
  NeuronIdx, I, Count, PinnedCount: integer;
  W: TNNetVolume;
  AbsW, MaxAbs, SumAbs, SumSquared, Mean, Variance, Stddev,
    CapThreshold: TNeuralFloat;
begin
  if Layer = nil then exit;
  MaxAbs := 0;
  SumAbs := 0;
  SumSquared := 0;
  Count := 0;
  PinnedCount := 0;
  CapThreshold := ClipMax * 0.99;

  for NeuronIdx := 0 to Layer.Neurons.Count - 1 do
  begin
    W := Layer.Neurons[NeuronIdx].Weights;
    if W = nil then continue;
    for I := 0 to W.Size - 1 do
    begin
      AbsW := Abs(W.FData[I]);
      if AbsW > MaxAbs then MaxAbs := AbsW;
      SumAbs := SumAbs + AbsW;
      SumSquared := SumSquared + AbsW * AbsW;
      Inc(Count);
      if AbsW >= CapThreshold then Inc(PinnedCount);
    end;
  end;

  if Count = 0 then exit;

  Mean := SumAbs / Count;
  Variance := SumSquared / Count - Mean * Mean;
  if Variance < 0 then Variance := 0;
  Stddev := Sqrt(Variance);

  WriteLn(Format(
    '  [W] %s max=%.4f mean=%.4f std=%.4f pinned=%d/%d (%.1f%%)',
    [LayerName, MaxAbs, Mean, Stddev, PinnedCount, Count,
     100.0 * PinnedCount / Count]));
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
  ShouldDumpStats: boolean;
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

    // Diagnostic dump at the same cadence as the training log. Lets weight
    // evolution be correlated with accuracy across the run. Tracks the
    // peak-then-collapse hypothesis: if homogenization-via-migration is
    // happening, std|W| collapses toward 0 and pinned% climbs toward 100
    // for Q/K projections in the collapse window. If something else is
    // driving the collapse, these stay healthy and we look elsewhere.
    ShouldDumpStats := (Sender = FNFit) and (FNFit.LogEveryBatches > 0)
      and ((FNFit.CurrentStep + 1) mod FNFit.LogEveryBatches = 0);
    if ShouldDumpStats then
    begin
      for AttIdx := 0 to FNN.AttentionLayers.Count - 1 do
      begin
        Info := TKANAttentionLayerInfo(FNN.AttentionLayers[AttIdx]);
        if Info.QProjection <> nil then
          DumpLayerWeightStats(Info.QProjection,
            Format('Att%d.Q', [AttIdx]), csQKWeightClipMax);
        if Info.KProjection <> nil then
          DumpLayerWeightStats(Info.KProjection,
            Format('Att%d.K', [AttIdx]), csQKWeightClipMax);
      end;
    end;
  finally
    QKVLayers.Free;
  end;
end;

end.
