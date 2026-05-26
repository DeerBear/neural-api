unit kantransformerarch;

(*
KAN transformer architecture construction.

Pure-construction unit: builds the TKANNet object and returns it. No
training, no inference, no dataset awareness. The architecture is
identical between training and inference programs -- shared via this
unit so neither program duplicates layer specs.

The architecture mirrors TNNet.AddTransformerBlockCAI(Heads=16,
IntermediateDim=512, pActFn=TNNetSignedSquareRoot1) but swaps the per-
head softmax in the attention sub-chain for TNNetKANNormaliser via
TKANNet.AddKANSelfAttention. ~1.39M total parameters, with ~525K per
transformer block.
*)

interface

uses
  neuralnetwork, neuralvolume, neuralkanattention;

const
  // Input shape and tokenisation
  csContextLen        = 81;      // Legacy default. Prefer passing ContextLen
                                 // explicitly to BuildKANTransformer1M, sourced
                                 // from TKANTransformerDataset.RecommendedContextLen
                                 // (log-log mean of line lengths).
  csVocabSize         = 128;     // 7-bit ASCII char-level vocabulary

  // Embedding / sequence-mixing conv
  csConvOutChannels   = 32;      // Per-position ReLU conv depth
  csEmbeddingDim      = 256;     // Sequence-mixing conv output channels
  csConvKernel        = 13;      // Sequence-mixing kernel + stride. Attention
                                 // position count = ContextLen div csConvKernel
                                 // (e.g. 81/13=6, 200/13=15). Attention scales
                                 // O(N^2) in position count; doubling ContextLen
                                 // ~quadruples per-layer attention cost.

  // Transformer block geometry
  csTransformerBlocks = 2;
  csHeads             = 16;      // Active heads per attention block; the
                                 // ceiling is the same in v1 since head
                                 // squaring (S4) is not yet implemented.
  csHeadCeiling       = 16;
  csFFNIntermediate   = 512;     // PointwiseConvReLU expand size

  // Output head
  csHiddenLayerSize   = 128;     // FullConnectReLU before vocab head

  // Total params: ~1.39M at ContextLen=81; check via FNN.CountWeights
  // after construction.

function BuildKANTransformer1M(ContextLen: integer = csContextLen): TKANNet;

implementation

function BuildKANTransformer1M(ContextLen: integer): TKANNet;
var
  I, EmbeddingDim: integer;
  PrevLayer, Attended, AttendedPlusPrev: TNNetLayer;
begin
  Result := TKANNet.Create();
  Result.AddLayer([
    TNNetInput.Create(ContextLen, 1, csVocabSize),
    TNNetAddPositionalEmbedding.Create(10000),
    TNNetConvolutionReLU.Create(csConvOutChannels, 1, 0, 1, 0),
    TNNetConvolution.Create(csEmbeddingDim, csConvKernel, 0, csConvKernel, 0)
  ]);

  for I := 1 to csTransformerBlocks do
  begin
    PrevLayer := Result.GetLastLayer();
    EmbeddingDim := PrevLayer.Output.Depth;
    Attended := Result.AddKANSelfAttention(csHeads, csHeadCeiling);
    AttendedPlusPrev := Result.AddLayer( TNNetSum.Create([Attended, PrevLayer]) );
    AttendedPlusPrev := Result.AddLayer( TNNetSignedSquareRoot1.Create() );
    Result.AddLayer( TNNetPointwiseConvReLU.Create(csFFNIntermediate, 1) );
    Result.AddLayer( TNNetSignedSquareRoot1.Create() );
    Result.AddLayer( TNNetPointwiseConvLinear.Create(EmbeddingDim, 1) );
    Result.AddLayer( TNNetSignedSquareRoot1.Create() );
    Result.AddLayer( TNNetSum.Create([Result.GetLastLayer(), AttendedPlusPrev]) );
    Result.AddLayer( TNNetSignedSquareRoot1.Create() );
  end;

  Result.AddLayer([
    TNNetFullConnectReLU.Create(csHiddenLayerSize),
    TNNetFullConnectReLU.Create(csVocabSize),
    TNNetSoftMax.Create()
  ]);
end;

end.
