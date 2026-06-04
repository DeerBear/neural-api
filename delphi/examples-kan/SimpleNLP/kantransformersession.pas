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
  kantransformerdata, mmaptextdataset, prefetchloader, checkpointcompanion;

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

  // Whether the energy-conserving clip runs on the *content* path
  // (embedding, FFN, attention output projections, vocab head) at all.
  //
  // Epoch-3 weight dumps settled this: attention never reaches its own
  // cap (Q/K/V max ~0.11 vs the 0.40 cap, 0% pinned), so the clip is a
  // dormant guardrail there. But the network-wide 0.20 content clip was
  // actively damaging the content path -- the embedding (L2) leaked to
  // max 0.42 with 11% pinned (a single-pass spread that never reconverges,
  // confirmed with momentum off), and the vocab head (L1319) had 4.5% of
  // its weights pinned at the cap, shaving exactly the peaked weights that
  // let rare tokens surface. Distribution shaping is the KAN's job, not
  // the clip's: the clip stays neutral hygiene on the attention
  // projections, and the embedding / FFN / output head learn unconstrained.
  // Set True to restore the old network-wide clip.
  csClipContentLayers: boolean = false;

  // Looser threshold for Q/K/V projection layers in attention blocks.
  // Attention dot products scale as Q*K, so capping Q/K at the same
  // magnitude as the rest of the network (0.20) keeps softmax outputs
  // near uniform 1/H = degenerate attention. The previous v1.1 run hit
  // exactly this failure: Q/K weights settled around 0.11, dot products
  // landed at 0.03-0.10, softmax outputs at 0.164-0.169 (uniform = 0.167).
  // 0.40 gives the projections room to grow into a regime where dot
  // products are meaningful (target: 0.5-2.0 range after MulByConstant).
  csQKWeightClipMax: TNeuralFloat = 0.40;

  // Override for TNeuralDataLoadingFit.MaxThreadNum. Without this set,
  // the framework picks NeuralDefaultThreadCount() == TThread.ProcessorCount
  // (4 on the reference 4-thread non-AVX i5). Bumping to 6 gives 1.5x
  // oversubscription -- the OS scheduler handles it, framework does not
  // cap MaxThreadNum, and memory cost is one extra TNet clone (~5.5 MB).
  // Tunable for different host machines; revert to 4 to match the default
  // exactly, or push higher (8, 12...) on bigger CPUs.
  csTrainingThreadCount: integer = 6;

  // Adaptive controller (epoch-boundary triggers + adjustments).
  //
  // At every epoch boundary past the first, the session inspects:
  //   * The per-log-boundary TrainingAccuracy history collected during
  //     the epoch (oscillation = max - min). Above the threshold signals
  //     "training destabilized" and unlocks the adjustment selector.
  //   * The most recent Q/K weight stats. These pick which adjustment
  //     matches the failure mode:
  //       - Any layer with pinned% >= csQKPinnedTriggerPct   -> migration cap
  //       - Any layer with max >= csQKMaxNearCapPct% of cap
  //         AND pinned% < csQKLowPinnedThresholdPct          -> tighten cap
  //
  // Adjustments are sticky (once on, stay on) and cumulative across
  // epochs -- so a homogenization-then-saturation sequence can apply
  // both fixes. Whichever adjustment kicks in is, by construction, the
  // one that matched the actual pathology; subsequent epoch accuracy is
  // ground-truth confirmation of (or pushback on) that diagnosis.
  csAccuracyOscillationThreshold: TNeuralFloat = 0.05;
  csQKPinnedTriggerPct: TNeuralFloat = 25.0;
  csQKMaxNearCapPct: TNeuralFloat = 99.0;
  csQKLowPinnedThresholdPct: TNeuralFloat = 10.0;

  // Per-neuron migration-cap trigger. Inside ClipAndSpreadWeights, when
  // the migration cap is active, any neuron whose pre-clip weights are
  // already this fraction pinned at the cap will drop the excess rather
  // than redistribute it. Breaks the "weight grows -> hits cap -> its
  // excess feeds the next weight -> that one grows toward cap -> ..."
  // migration loop that drives homogenization of the entire neuron.
  csMigrationCapPerNeuronPinnedPct: TNeuralFloat = 25.0;

  // Multiplicative scale applied to the runtime Q/K clip cap when the
  // saturation trigger fires (0.75 takes 0.40 -> 0.30 -> 0.225 ...).
  csTightenCapFactor: TNeuralFloat = 0.75;

  // Number of log-boundary samples retained per epoch for the
  // accuracy-oscillation diagnostic.
  csAccuracyHistorySize: integer = 16;

  // Plateau-triggered ActiveHeads doubling. The architecture builds
  // csHeadCeiling head slots (currently 64) and starts with csHeads
  // active (currently 8). When validation loss fails to improve for
  // csHeadDoubleWindow consecutive epochs, ActiveHeads doubles
  // (8 -> 16 -> 32 -> 64) on every attention layer. Independent of the
  // longer csPlateauWindow (10) that triggers training stop -- doubling
  // is a corrective response, stopping is a termination response.
  csHeadDoubleWindow: integer = 2;

  // ReduceLROnPlateau: when validation loss fails to beat its best for
  // csLRPatience consecutive epochs, multiply the learning rate by csLRFactor,
  // down to a floor of csLRFloorFraction * the initial rate. Targets the
  // noisy-constant-LR failure mode; fully independent of the weight clip.
  csLRPatience: integer = 1;
  csLRFactor: TNeuralFloat = 0.7;
  csLRFloorFraction: TNeuralFloat = 0.01;

type
  // Snapshot of a layer's weight distribution at one moment in time.
  // Returned by ComputeLayerWeightStats; consumed by the adaptive
  // controller's adjustment selector at the epoch boundary.
  //
  // L2: sqrt(Sum(W_i^2)). The full-energy invariant the clipper
  //   preserves. Drifts even when max/mean/std are frozen at 4dp,
  //   because rearrangement among individual weights changes it.
  // ActMax / ActMin: max(|out|) and signed min(out) of the layer's
  //   FOutput right after the most recent forward pass. Tells us
  //   what magnitudes the layer is producing, not just what
  //   magnitudes the weights have.
  TLayerWeightStats = record
    MaxAbs: TNeuralFloat;
    Mean: TNeuralFloat;
    Stddev: TNeuralFloat;
    PinnedPct: TNeuralFloat;  // [0, 100]
    L2: TNeuralFloat;
    ActMax: TNeuralFloat;
    ActMin: TNeuralFloat;
    Count: integer;
  end;

  TKANTransformerSession = class
  private
    FNN: TKANNet;
    FDataset: TKANTransformerDataset;
    FNFit: TNeuralDataLoadingFit;
    FSampler: TNNetSamplerBase;

    // Memory-mapped training source + async prefetch -- the "load once, work
    // downstream" experiment. FUseMmapTraining routes the training hot loop
    // through the mmap loader; FUsePrefetch additionally decouples sample prep
    // onto a background loader thread. Validation/test stay on the eager
    // FDataset. Both default on; flip them (and MaxThreadNum) for the A/B.
    FMapped: TMappedTextDataset;
    FPrefetcher: TPrefetcher;
    FUseMmapTraining: boolean;
    FUsePrefetch: boolean;

    // Companion-file state -- the trainer is authoritative for what a checkpoint
    // is. At Train start it captures the context the network was actually built
    // at and hashes the corpus once (the corpus does not change mid-run); on
    // every epoch, after the framework has saved the .nn, OnAfterEpoch writes
    // the companion describing that just-saved checkpoint. Inference only reads.
    FCompContextSize: integer;
    FCompCorpusMD5: string;

    // ReduceLROnPlateau state (initialised in Train).
    FAdaptiveLR: TNeuralFloat;
    FLRStaleEpochs: integer;
    FLRBestLoss: TNeuralFloat;
    // Plateau-based early stopping: track best ValidationLoss and the
    // epoch it was achieved. If FPlateauWindow epochs pass without a
    // new record, the run is treated as converged and ShouldQuit is
    // signalled.
    FBestLoss: TNeuralFloat;
    FBestLossEpoch: integer;
    FPlateauWindow: integer;

    // Adaptive controller state. FQKClipMax starts at csQKWeightClipMax
    // and gets scaled down by csTightenCapFactor each time the
    // saturation trigger fires; FMigrationCapEnabled toggles per-neuron
    // migration-cap logic in ClipAndSpreadWeights when the
    // homogenization trigger fires. Both adjustments are sticky.
    FQKClipMax: TNeuralFloat;
    FMigrationCapEnabled: boolean;

    // Plateau-triggered head doubling. Tracks best ValidationLoss
    // independently of FBestLoss/FBestLossEpoch (which feed the
    // separate, longer stop-condition plateau). FHeadDoubleBestEpoch
    // is reset every time ActiveHeads is bumped so the next doubling
    // is gated by csHeadDoubleWindow fresh epochs of no improvement.
    FHeadDoubleBestLoss: TNeuralFloat;
    FHeadDoubleBestEpoch: integer;

    // Per-epoch trackers, reset at OnAfterEpoch. FAccuracyHistory is a
    // ring buffer of TrainingAccuracy sampled at each log boundary;
    // FLastQKStats holds the most recent Q/K stats so the trigger
    // evaluator can read them without recomputing.
    FAccuracyHistory: array of TNeuralFloat;
    FAccuracyHistoryIdx: integer;
    FAccuracyHistoryCount: integer;
    FLastQKStats: array of TLayerWeightStats;
    FLastQKLayerNames: array of string;

    procedure OnAfterEpoch(Sender: TObject);
    procedure OnAfterStep(Sender: TObject);
    procedure GenerateSamples;
    /// Iterate FNN.AttentionLayers, call DoubleActiveHeads on each, and
    /// log a single summary line. Invoked from OnAfterEpoch when the
    /// head-double plateau condition fires.
    procedure MaybeDoubleAllActiveHeads;
    /// Training hot-loop getter. Routes to the prefetcher, the direct mmap
    /// loader, or the eager dataset depending on the FUse* flags.
    procedure GetTrainingPairRouted(Idx, ThreadId: integer;
      pInput, pOutput: TNNetVolume);
    /// Deterministic-by-Idx validation/test getter, served from the single
    /// mmap source. Used for FitLoading val+test, CalibrateAlpha, and eval.
    procedure GetValidationPairRouted(Idx, ThreadId: integer;
      pInput, pOutput: TNNetVolume);
    /// ReduceLROnPlateau callback: returns the current adaptive learning rate
    /// (lowered by OnAfterEpoch when validation stalls). Assigned to the fit's
    /// CustomLearningRateScheduleObjFn.
    function LRSchedule(Epoch: integer): single;
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
    /// Teacher-forced, frequency-stratified next-char NLL on held-out data,
    /// no sampler involved. Run on the SAME checkpoint in both modes (KAN
    /// disabled = softmax baseline, vs KAN engaged+warmed) and compare the
    /// 'rare' third -- the sampler-free measurement of whether KAN puts more
    /// probability mass on rare-but-correct tokens. Scores whatever mode the
    /// network is currently in; the caller sets the mode.
    procedure EvaluateStratifiedNLL(const SampleCount: integer);
    property PlateauWindow: integer
      read FPlateauWindow write FPlateauWindow;
    // Route training through the mmap loader (default true). When false, the
    // eager FDataset.GetTrainingPair is used -- the baseline arm of the A/B.
    property UseMmapTraining: boolean
      read FUseMmapTraining write FUseMmapTraining;
    // Decouple sample prep onto a background loader thread (default true).
    // With this on, the page faults live in the loader, so keep compute
    // threads at core count. With it off (workers read the mmap directly),
    // the workers fault -- that's the over-subscribe-the-workers arm.
    property UsePrefetch: boolean
      read FUsePrefetch write FUsePrefetch;
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
procedure ClipAndSpreadWeights(W: TNNetVolume; const MaxAbs: TNeuralFloat;
  const MigrationCapEnabled: boolean = false);
var
  I, PreClipPinnedCount: integer;
  Weight, AbsWeight, TotalExcessEnergy, UnclippedEnergy, EnergyFactor,
    CapThreshold, PinnedPct: TNeuralFloat;
  IsClipped: array of boolean;
  DropExcess: boolean;
begin
  if (W = nil) or (W.Size = 0) or (MaxAbs <= 0) then exit;

  // Migration cap: if the neuron's weights are already over the pinned%
  // threshold before this step's clipping pass, the redistribution loop
  // would just feed more weights into the clipped set. Drop the excess
  // instead. Disabled by default; the adaptive controller flips this on
  // when the homogenization trigger fires at an epoch boundary.
  DropExcess := false;
  if MigrationCapEnabled then
  begin
    CapThreshold := MaxAbs * 0.99;
    PreClipPinnedCount := 0;
    for I := 0 to W.Size - 1 do
      if Abs(W.FData[I]) >= CapThreshold then Inc(PreClipPinnedCount);
    PinnedPct := 100.0 * PreClipPinnedCount / W.Size;
    DropExcess := PinnedPct >= csMigrationCapPerNeuronPinnedPct;
  end;

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
  if DropExcess then exit;

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

// Compute max|W|, mean|W|, std|W|, and pinned% (fraction of weights at
// or above 99% of the cap) for one layer, across every weight in every
// neuron. Returns a TLayerWeightStats; does not print. The four numbers
// together distinguish:
//   * Healthy:      max < cap, std non-trivial, pinned% ~ 0
//   * Stuck small:  max << cap, std small, pinned% = 0
//   * Saturating:   max == cap, std non-trivial, pinned% low but rising
//   * Homogenized:  max == cap, std collapsing, pinned% climbing -> 100
// The collapse signature we're hunting is "homogenized" -- std
// collapsing while pinned% rises -- indicating weights migrating into
// the clipped subset and being held there by the clipper.
function ComputeLayerWeightStats(Layer: TNNetLayer;
  const ClipMax: TNeuralFloat): TLayerWeightStats;
var
  NeuronIdx, I, PinnedCount: integer;
  W: TNNetVolume;
  AbsW, SumAbs, SumSquared, Variance, CapThreshold: TNeuralFloat;
begin
  Result.MaxAbs := 0;
  Result.Mean := 0;
  Result.Stddev := 0;
  Result.PinnedPct := 0;
  Result.L2 := 0;
  Result.ActMax := 0;
  Result.ActMin := 0;
  Result.Count := 0;
  if Layer = nil then exit;

  SumAbs := 0;
  SumSquared := 0;
  PinnedCount := 0;
  CapThreshold := ClipMax * 0.99;

  for NeuronIdx := 0 to Layer.Neurons.Count - 1 do
  begin
    W := Layer.Neurons[NeuronIdx].Weights;
    if W = nil then continue;
    for I := 0 to W.Size - 1 do
    begin
      AbsW := Abs(W.FData[I]);
      if AbsW > Result.MaxAbs then Result.MaxAbs := AbsW;
      SumAbs := SumAbs + AbsW;
      SumSquared := SumSquared + AbsW * AbsW;
      Inc(Result.Count);
      if AbsW >= CapThreshold then Inc(PinnedCount);
    end;
  end;

  if Result.Count = 0 then exit;

  Result.Mean := SumAbs / Result.Count;
  Variance := SumSquared / Result.Count - Result.Mean * Result.Mean;
  if Variance < 0 then Variance := 0;
  Result.Stddev := Sqrt(Variance);
  Result.PinnedPct := 100.0 * PinnedCount / Result.Count;
  Result.L2 := Sqrt(SumSquared);

  // Activation magnitudes from the layer's FOutput (snapshot from the
  // most recent forward pass). Useful for spotting saturation, dead
  // ReLUs, and runaway activations that wouldn't show up in weight
  // stats. Layer.Output is non-nil for every layer with FOutput set up;
  // we still guard against an empty volume defensively.
  if (Layer.Output <> nil) and (Layer.Output.Size > 0) then
  begin
    Result.ActMax := Layer.Output.GetMaxAbs();
    Result.ActMin := Layer.Output.GetMin();
  end;
end;

procedure PrintLayerWeightStats(const LayerName: string;
  const Stats: TLayerWeightStats);
begin
  if Stats.Count = 0 then exit;
  // Activation columns (out |max|, out min) intentionally not printed:
  // the framework uses TNNetDataParallelism, so each thread does its
  // forward pass into a clone's FOutput. The main FNN we have a handle
  // to never sees populated activations -- they'd always read 0.0000.
  // Weight stats sync back to FNN after each batch and are correct.
  WriteLn(Format(
    '  [W] %-14s w max=%.4f mean=%.4f std=%.4f L2=%.3f pin=%.1f%%',
    [LayerName, Stats.MaxAbs, Stats.Mean, Stats.Stddev, Stats.L2,
     Stats.PinnedPct]));
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

  // One persistent memory-mapped data source for the whole session lifetime
  // -- training (random), validation, and test all read from this single
  // mapping, so the corpus is resident exactly once (~2 GB, not ~4 GB). Built
  // here (not in Train) so post-training eval/calibration still have data.
  FMapped := TMappedTextDataset.Create(FNN.GetFirstLayer().Output.SizeX);
  FMapped.LoadDataset(FDataset.FileName);
  FMapped.BindNetwork(FNN);
  FPrefetcher := nil;
  FUseMmapTraining := true;
  FUsePrefetch := true;

  FQKClipMax := csQKWeightClipMax;
  FMigrationCapEnabled := false;
  FHeadDoubleBestLoss := 1e30;
  FHeadDoubleBestEpoch := 0;
  SetLength(FAccuracyHistory, csAccuracyHistorySize);
  FAccuracyHistoryIdx := 0;
  FAccuracyHistoryCount := 0;
  // FLastQKStats / FLastQKLayerNames sized on first OnAfterStep dump.
end;

destructor TKANTransformerSession.Destroy;
begin
  // Defensive: Train frees these in its finally block, but tear them down
  // here too in case Train was never called or raised before cleanup.
  if Assigned(FPrefetcher) then
  begin
    FPrefetcher.Stop;
    FreeAndNil(FPrefetcher);
  end;
  FreeAndNil(FMapped);
  FSampler.Free;
  FNFit.Free;
  inherited Destroy;
end;

procedure TKANTransformerSession.GetTrainingPairRouted(Idx, ThreadId: integer;
  pInput, pOutput: TNNetVolume);
begin
  // With prefetch on, pop a sample the background loader already built from
  // the mmap (workers stay CPU-bound, faults live in the loader). With it
  // off but mmap on, read the mapping directly (workers fault -- the
  // oversubscription-hiding arm). Otherwise fall back to the eager dataset.
  // Prefetch on: pop a sample the loader already built from the mmap. Off:
  // read the mapping directly (workers fault -- the oversubscription arm).
  if FUsePrefetch and Assigned(FPrefetcher) then
    FPrefetcher.GetPair(pInput, pOutput)
  else
    FMapped.BuildTrainingSample(pInput, pOutput);
end;

procedure TKANTransformerSession.GetValidationPairRouted(Idx, ThreadId: integer;
  pInput, pOutput: TNNetVolume);
begin
  FMapped.BuildValidationSample(Idx, pInput, pOutput);
end;

function TKANTransformerSession.LRSchedule(Epoch: integer): single;
begin
  // OnAfterEpoch lowers FAdaptiveLR on stalled validation; this just hands the
  // framework the current value each epoch.
  Result := FAdaptiveLR;
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
  // Override the framework's auto-detected thread count. See
  // csTrainingThreadCount in the const block above for rationale.
  FNFit.MaxThreadNum := csTrainingThreadCount;
  FNFit.OnAfterEpoch := OnAfterEpoch;
  FNFit.OnAfterStep := OnAfterStep;

  // ReduceLROnPlateau: drive the LR from FAdaptiveLR, which OnAfterEpoch lowers
  // when validation stalls. Starts at the configured initial rate.
  FAdaptiveLR := FNFit.InitialLearningRate;
  FLRStaleEpochs := 0;
  FLRBestLoss := 1e30;
  FNFit.CustomLearningRateScheduleObjFn := LRSchedule;

  // Trainer-authoritative companion setup. Capture the context the network was
  // actually built at (its input layer's X size) and hash the corpus once;
  // both are reused when OnAfterEpoch writes the companion after each save.
  FCompContextSize := FNN.GetFirstLayer().Output.SizeX;
  // Corpus hash = the dataset's chained HMAC-MD5 over the raw lines, computed
  // during LoadDataset (already run before Train) -- not a plain file MD5.
  FCompCorpusMD5 := FDataset.CorpusHash;
  WriteLn(Format('  Companion: context=%d, corpus hash=%s',
    [FCompContextSize, FCompCorpusMD5]));

  // Spin up the prefetch loader over the persistent single mmap source
  // (created in the constructor). FMapped is not Train-scoped any more.
  if FUseMmapTraining and FUsePrefetch then
  begin
    FPrefetcher := TPrefetcher.Create(FMapped.BuildTrainingSample);
    FPrefetcher.Start;
  end;

  try
    FNFit.FitLoading(
      FNN,
      TrainingCount,
      ValidationCount,
      TestCount,
      BatchSize,
      Epochs,
      GetTrainingPairRouted,
      GetValidationPairRouted,
      GetValidationPairRouted
    );
  finally
    if Assigned(FPrefetcher) then
    begin
      FPrefetcher.Stop;
      FreeAndNil(FPrefetcher);
    end;
  end;
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
  FNN.CalibrateAlpha(GetValidationPairRouted, ValidationCount);
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

procedure TKANTransformerSession.MaybeDoubleAllActiveHeads;
var
  I, BeforeActive, AfterActive, HMax: integer;
  Info: TKANAttentionLayerInfo;
  Doubled: boolean;
  Summary: string;
begin
  Doubled := false;
  Summary := '';
  for I := 0 to FNN.AttentionLayers.Count - 1 do
  begin
    Info := TKANAttentionLayerInfo(FNN.AttentionLayers[I]);
    BeforeActive := Info.ActiveHeads;
    HMax := Info.HMax;
    if Info.DoubleActiveHeads then
    begin
      Doubled := true;
      AfterActive := Info.ActiveHeads;
      Summary := Summary + Format(' L%d:%d->%d', [I, BeforeActive, AfterActive]);
    end
    else
      Summary := Summary + Format(' L%d:%d(cap=%d)', [I, BeforeActive, HMax]);
  end;
  if Doubled then
    WriteLn(Format(
      '[Adaptive] Plateau %d epochs without ValidationLoss improvement; doubled ActiveHeads:%s',
      [csHeadDoubleWindow, Summary]))
  else
    WriteLn(Format(
      '[Adaptive] Plateau %d epochs without ValidationLoss improvement; ActiveHeads already at HMax on all layers (%s).',
      [csHeadDoubleWindow, Summary]));
end;

procedure TKANTransformerSession.OnAfterEpoch(Sender: TObject);
var
  I: integer;
  MinAcc, MaxAcc, Oscillation: TNeuralFloat;
  AnyHomogenizationMatch, AnySaturationMatch: boolean;
  Stats: TLayerWeightStats;
  Companion: TCheckpointCompanion;
begin
  GenerateSamples;

  // Plateau check. Only meaningful inside the training loop (FNFit
  // populates ValidationLoss / CurrentEpoch); skip otherwise.
  if Sender = FNFit then
  begin
    // Trainer-authoritative companion. OnAfterEpoch runs AFTER the framework
    // saves the .nn (the save precedes this callback), so the file on disk is
    // the just-saved checkpoint. (Re)write its companion -- context + the
    // once-hashed corpus MD5 + the .nn's current MD5 -- so it always describes
    // the checkpoint as saved. Skipped until a .nn exists.
    if FileExists(FNFit.FileNameBase + '.nn') then
    begin
      Companion := TCheckpointCompanion.Create(FNFit.FileNameBase + '.nn');
      try
        Companion.Write(FCompContextSize, FCompCorpusMD5);
      finally
        Companion.Free;
      end;
    end;

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

    // ReduceLROnPlateau: independent of the stop / head-double plateaus. When
    // validation fails to beat its best for csLRPatience epochs, scale the
    // adaptive LR down (to the floor) and reset the counter. OnAfterEpoch runs
    // at epoch end; LRSchedule hands the new value to the next epoch.
    if FNFit.ValidationLoss < FLRBestLoss then
    begin
      FLRBestLoss := FNFit.ValidationLoss;
      FLRStaleEpochs := 0;
    end
    else
    begin
      Inc(FLRStaleEpochs);
      if FLRStaleEpochs >= csLRPatience then
      begin
        if FAdaptiveLR * csLRFactor >=
           FNFit.InitialLearningRate * csLRFloorFraction then
        begin
          FAdaptiveLR := FAdaptiveLR * csLRFactor;
          WriteLn(Format(
            '[Adaptive] ValidationLoss stalled %d epoch(s); learning rate -> %.6f',
            [FLRStaleEpochs, FAdaptiveLR]));
        end
        else
          WriteLn('[Adaptive] ValidationLoss stalled but LR at floor; holding.');
        FLRStaleEpochs := 0;
      end;
    end;

    // Plateau-triggered head doubling. Tracked on a separate, shorter
    // window than the stop-condition plateau above. When
    // csHeadDoubleWindow epochs pass with no validation-loss
    // improvement, every attention layer doubles its ActiveHeads (up
    // to its HMax). The doubling resets the head-double-plateau
    // counter; the stop plateau counter is unaffected.
    if FNFit.ValidationLoss < FHeadDoubleBestLoss then
    begin
      FHeadDoubleBestLoss := FNFit.ValidationLoss;
      FHeadDoubleBestEpoch := FNFit.CurrentEpoch;
    end
    else if (FNFit.CurrentEpoch - FHeadDoubleBestEpoch) >= csHeadDoubleWindow then
    begin
      MaybeDoubleAllActiveHeads;
      FHeadDoubleBestEpoch := FNFit.CurrentEpoch;
    end;

    // Adaptive controller. Past epoch 1 only (initial-fit dynamics
    // dominate epoch 0/1 oscillation and aren't the destabilization
    // we're trying to catch). Trigger evaluation reads the ring buffer
    // populated by OnAfterStep, then -- if destabilized -- picks an
    // adjustment by matching the latest Q/K weight stats against the
    // homogenization / saturation signatures.
    if (FNFit.CurrentEpoch > 1) and (FAccuracyHistoryCount > 1) then
    begin
      MinAcc := FAccuracyHistory[0];
      MaxAcc := FAccuracyHistory[0];
      for I := 1 to FAccuracyHistoryCount - 1 do
      begin
        if FAccuracyHistory[I] < MinAcc then MinAcc := FAccuracyHistory[I];
        if FAccuracyHistory[I] > MaxAcc then MaxAcc := FAccuracyHistory[I];
      end;
      Oscillation := MaxAcc - MinAcc;

      WriteLn(Format(
        '[Adaptive] Epoch %d end. Accuracy oscillation: %.4f over %d samples. State: MigrationCap=%s, QKClipMax=%.4f',
        [FNFit.CurrentEpoch, Oscillation, FAccuracyHistoryCount,
         BoolToStr(FMigrationCapEnabled, true), FQKClipMax]));

      if Oscillation > csAccuracyOscillationThreshold then
      begin
        WriteLn('[Adaptive] Training destabilization detected (oscillation > threshold).');

        AnyHomogenizationMatch := false;
        AnySaturationMatch := false;
        for I := 0 to Length(FLastQKStats) - 1 do
        begin
          Stats := FLastQKStats[I];
          if Stats.Count = 0 then continue;
          if Stats.PinnedPct >= csQKPinnedTriggerPct then
            AnyHomogenizationMatch := true;
          if (Stats.MaxAbs >= FQKClipMax * csQKMaxNearCapPct / 100.0)
             and (Stats.PinnedPct < csQKLowPinnedThresholdPct) then
            AnySaturationMatch := true;
        end;

        // Adjustments are mutually exclusive within one epoch (the
        // homogenization fix takes precedence) but sticky across
        // epochs. If migration cap is already on and the signature
        // still matches, we don't re-apply -- the next epoch will
        // either show recovery or escalate to saturation.
        if AnyHomogenizationMatch and not FMigrationCapEnabled then
        begin
          FMigrationCapEnabled := true;
          WriteLn(Format(
            '[Adaptive] Homogenization signature matched (pinned%% >= %.1f on a Q/K layer). Enabling per-neuron migration cap from next epoch.',
            [csQKPinnedTriggerPct]));
        end
        else if AnySaturationMatch then
        begin
          FQKClipMax := FQKClipMax * csTightenCapFactor;
          WriteLn(Format(
            '[Adaptive] Saturation signature matched (max ~= cap, pinned%% low). Tightening Q/K clip cap to %.4f from next epoch.',
            [FQKClipMax]));
        end
        else if AnyHomogenizationMatch then
          WriteLn('[Adaptive] Homogenization signature still present; migration cap already enabled, no further action this epoch.')
        else
          WriteLn('[Adaptive] Destabilized but no matching Q/K weight signature; no auto-fix applied. Look at FFN / optimizer / gradient norms.');
      end;
    end;

    // Reset ring buffer for the next epoch. Adjustments persist; the
    // diagnostic data does not.
    FAccuracyHistoryCount := 0;
    FAccuracyHistoryIdx := 0;
  end;
end;

procedure TKANTransformerSession.OnAfterStep(Sender: TObject);
var
  LayerIdx, NeuronIdx, AttIdx, QKSlot, QKCount: integer;
  Layer: TNNetLayer;
  Info: TKANAttentionLayerInfo;
  QKVLayers: TList;
  ClipMax: TNeuralFloat;
  IsQKVLayer, ShouldDumpStats: boolean;
  LayerName: string;
  Stats: TLayerWeightStats;
begin
  // Apply energy-conserving weight clipping per-neuron across the entire
  // network after each batch. Per-neuron (rather than per-layer) preserves
  // the relative weight budgets between output channels; each channel's
  // weights spread among themselves but channels stay independent.
  //
  // Two thresholds: FQKClipMax (runtime, starts at csQKWeightClipMax,
  // adaptively tightened on saturation triggers) for Q/K/V projection
  // layers in attention blocks; csWeightClipMax for everything else.
  // Migration cap is passed only when the homogenization trigger has
  // fired on a prior epoch.
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
      IsQKVLayer := QKVLayers.IndexOf(Layer) >= 0;
      // Attention projections always clip (at the looser Q/K/V cap, where
      // it acts as a dormant guardrail). The content path clips only when
      // csClipContentLayers is set -- by default it learns unconstrained,
      // so the embedding stops leaking and the vocab head stops pinning.
      if IsQKVLayer then
        ClipMax := FQKClipMax
      else if csClipContentLayers then
        ClipMax := csWeightClipMax
      else
        continue;
      for NeuronIdx := 0 to Layer.Neurons.Count - 1 do
        ClipAndSpreadWeights(Layer.Neurons[NeuronIdx].Weights, ClipMax,
          FMigrationCapEnabled);
    end;

    // Diagnostic dump + adaptive controller sampling, both at the same
    // cadence as the training log. The per-Q-and-K stats are printed
    // for visibility AND retained for the epoch-boundary trigger
    // evaluator; the current TrainingAccuracy goes into the ring buffer
    // so the same evaluator can read oscillation.
    ShouldDumpStats := (Sender = FNFit) and (FNFit.LogEveryBatches > 0)
      and ((FNFit.CurrentStep + 1) mod FNFit.LogEveryBatches = 0);
    if ShouldDumpStats then
    begin
      QKCount := 2 * FNN.AttentionLayers.Count;
      if Length(FLastQKStats) <> QKCount then
      begin
        SetLength(FLastQKStats, QKCount);
        SetLength(FLastQKLayerNames, QKCount);
      end;
      for AttIdx := 0 to FNN.AttentionLayers.Count - 1 do
      begin
        Info := TKANAttentionLayerInfo(FNN.AttentionLayers[AttIdx]);
        QKSlot := AttIdx * 2;
        if Info.QProjection <> nil then
        begin
          LayerName := Format('Att%d.Q', [AttIdx]);
          Stats := ComputeLayerWeightStats(Info.QProjection, FQKClipMax);
          PrintLayerWeightStats(LayerName, Stats);
          FLastQKStats[QKSlot] := Stats;
          FLastQKLayerNames[QKSlot] := LayerName;
        end;
        if Info.KProjection <> nil then
        begin
          LayerName := Format('Att%d.K', [AttIdx]);
          Stats := ComputeLayerWeightStats(Info.KProjection, FQKClipMax);
          PrintLayerWeightStats(LayerName, Stats);
          FLastQKStats[QKSlot + 1] := Stats;
          FLastQKLayerNames[QKSlot + 1] := LayerName;
        end;
        // V is printed for visibility only; it does not feed the adaptive
        // trigger evaluator. The homogenization / saturation signatures
        // are properties of Q/K (which drive softmax sharpness), not V
        // (which carries the actual content vectors). Cap reference is the
        // same Q/K/V cap used by the clipper.
        if Info.VProjection <> nil then
        begin
          LayerName := Format('Att%d.V', [AttIdx]);
          Stats := ComputeLayerWeightStats(Info.VProjection, FQKClipMax);
          PrintLayerWeightStats(LayerName, Stats);
        end;
      end;

      // Dump every other neuron-bearing layer too. Catches movement in
      // the FFN, embedding-feeder convs, attention output projections,
      // and the final FC stack -- the layers we couldn't see before
      // and which (if "FFN+V learns first" is right) are doing most of
      // the actual learning while Q/K stays inert. Skip layers already
      // dumped above (the Q/K/V triples). Cap reference for non-Q/K/V
      // is csWeightClipMax (the tighter 0.20 cap).
      for LayerIdx := 0 to FNN.CountLayers - 1 do
      begin
        Layer := FNN.Layers[LayerIdx];
        if Layer.Neurons.Count = 0 then continue;
        if QKVLayers.IndexOf(Layer) >= 0 then continue;
        LayerName := Format('L%-3d %s', [LayerIdx, Layer.ClassName]);
        Stats := ComputeLayerWeightStats(Layer, csWeightClipMax);
        PrintLayerWeightStats(LayerName, Stats);
      end;

      // Ring-buffer the current training accuracy for the
      // oscillation diagnostic. Use the same sample point as the
      // stats dump so the two diagnostics align in time.
      FAccuracyHistory[FAccuracyHistoryIdx] := FNFit.TrainingAccuracy;
      FAccuracyHistoryIdx := (FAccuracyHistoryIdx + 1) mod csAccuracyHistorySize;
      if FAccuracyHistoryCount < csAccuracyHistorySize then
        Inc(FAccuracyHistoryCount);
    end;
  finally
    QKVLayers.Free;
  end;
end;

procedure TKANTransformerSession.EvaluateStratifiedNLL(const SampleCount: integer);
// Teacher-forced, frequency-stratified next-char NLL over held-out validation
// samples -- no sampler. For each sample it reads the probability the model
// assigns to the TRUE next char, accumulates -ln() per char, then buckets the
// vocab by how often each char actually occurs (common / mid / rare third of
// the observed token mass) and reports mean NLL per bucket. Run twice on the
// SAME checkpoint -- once KAN-disabled (softmax baseline), once KAN-engaged --
// and compare the 'rare' third: that is the sampler-free "does KAN put more
// mass on rare-but-correct tokens" measurement. NOTE: in KAN mode each Compute
// mutates the spline state, so the caller should SnapshotAllCoeffs before /
// RestoreAllCoeffs after for a non-destructive read.
var
  InputVol, OutputVol: TNNetVolume;
  V, I, J, Tok, B, tmp: integer;
  Prob, NLLv: TNeuralFloat;
  SumNLL: array of Double;
  Cnt: array of Int64;
  Order: array of integer;
  BSumNLL: array[0..2] of Double;
  BCnt: array[0..2] of Int64;
  TotalCnt, RunCnt: Int64;
  LastLayer: TNNetLayer;
const
  QFloor = 1e-30;
  BucketName: array[0..2] of string = ('common', 'mid', 'rare');
begin
  V := FNN.GetLastLayer.Output.Size;          // vocab size (softmax width)
  SetLength(SumNLL, V);
  SetLength(Cnt, V);
  SetLength(Order, V);
  for I := 0 to V - 1 do begin SumNLL[I] := 0; Cnt[I] := 0; Order[I] := I; end;

  InputVol := TNNetVolume.Create;
  OutputVol := TNNetVolume.Create;
  try
    LastLayer := FNN.GetLastLayer;
    for I := 0 to SampleCount - 1 do
    begin
      FMapped.BuildValidationSample(I, InputVol, OutputVol);
      Tok := OutputVol.Tag;                    // true next-char index
      if (Tok < 0) or (Tok >= V) then Continue;
      FNN.Compute(InputVol);
      Prob := LastLayer.Output.FData[Tok];
      if Prob < QFloor then Prob := QFloor;
      NLLv := -Ln(Prob);
      SumNLL[Tok] := SumNLL[Tok] + NLLv;
      Cnt[Tok] := Cnt[Tok] + 1;
    end;

    // Rank chars by observed frequency, descending. V<=128, O(n^2) is fine.
    for I := 0 to V - 2 do
      for J := I + 1 to V - 1 do
        if Cnt[Order[J]] > Cnt[Order[I]] then
        begin tmp := Order[I]; Order[I] := Order[J]; Order[J] := tmp; end;

    TotalCnt := 0;
    for I := 0 to V - 1 do TotalCnt := TotalCnt + Cnt[I];

    // Walk common -> rare, splitting the observed token mass into thirds.
    for B := 0 to 2 do begin BSumNLL[B] := 0; BCnt[B] := 0; end;
    RunCnt := 0;
    for I := 0 to V - 1 do
    begin
      Tok := Order[I];
      if Cnt[Tok] = 0 then Continue;
      if RunCnt < TotalCnt div 3 then B := 0
      else if RunCnt < (2 * TotalCnt) div 3 then B := 1
      else B := 2;
      BSumNLL[B] := BSumNLL[B] + SumNLL[Tok];
      BCnt[B] := BCnt[B] + Cnt[Tok];
      RunCnt := RunCnt + Cnt[Tok];
    end;

    WriteLn(Format('Stratified NLL over %d samples (%d scored tokens):',
      [SampleCount, TotalCnt]));
    for B := 0 to 2 do
      if BCnt[B] > 0 then
        WriteLn(Format('  %-6s third: mean NLL = %.4f  (%d tokens)',
          [BucketName[B], BSumNLL[B] / BCnt[B], BCnt[B]]));
    Flush(Output);
  finally
    InputVol.Free;
    OutputVol.Free;
  end;
end;

end.
