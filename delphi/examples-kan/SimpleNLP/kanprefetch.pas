unit kanprefetch;

(*
Asynchronous sample prefetch for the KAN transformer training loop.

Splits data preparation (sample selection + one-hot encoding) off the
compute threads. A single background loader thread pre-builds (input,
output) volume pairs into a bounded ready-queue while the worker threads
train on already-prepared samples. The loader is prep-bound, not ALU-bound,
so it overlaps with the compute threads rather than competing with them for
cores -- the one place adding a thread above the core count actually pays
off (the extra thread spends its life waiting on prep, not on the FPU).

This is a pure SCHEDULING change: the per-sample build logic is byte-for-byte
the existing one, so it cannot alter the training maths -- only wall-clock.
The win is bounded by the fraction of step time currently spent in the
getter; if that fraction is tiny, so is the win. Measure it.

Determinism note: this does NOT change the determinism story relative to the
current code. The existing getter already samples with the global Random()
from N worker threads concurrently, so per-thread sample order is not
bit-stable today. Prefetch shifts which thread sees which sample, but the
training maths and the statistical sample distribution are identical. A
fully reproducible schedule (loader-private seeded RNG) is a v2 refinement.

STATUS: v1, written without a Delphi toolchain available. This is
concurrency code -- validate on a real run. Blocking/locking is delegated to
TThreadedQueue; clean shutdown goes through DoShutDown so no thread can hang
on a full/empty queue. Buffers are recycled through a free pool so steady
state does no per-sample heap allocation.
*)

interface

uses
  Classes, SysUtils, SyncObjs, Generics.Collections,
  neuralvolume;

type
  // Fills one freshly-selected training sample. Called ONLY from the single
  // loader thread during normal running (never concurrently), so it may use
  // the same global-Random sampling the inline getter already uses. Must
  // size the volumes itself, exactly as the existing GetTrainingPair does.
  TKANBuildSampleProc = procedure(Input, Output: TNNetVolume) of object;

  // One recycled buffer: a prepared (input, output) pair.
  TKANSamplePair = class
  public
    Input: TNNetVolume;
    Output: TNNetVolume;
    constructor Create;
    destructor Destroy; override;
  end;

  TKANPrefetcher = class
  private
    FReady: TThreadedQueue<TKANSamplePair>;   // built, waiting for a worker
    FFree:  TThreadedQueue<TKANSamplePair>;    // recycled, waiting for the loader
    FLoader: TThread;
    FBuild: TKANBuildSampleProc;
    FRunning: boolean;
    procedure ProduceLoop;
  public
    // ABuild: the dataset's per-sample builder. ADepth: ready-queue capacity
    // (a few batches' worth is plenty; this is also the buffer-pool size).
    constructor Create(const ABuild: TKANBuildSampleProc;
      const ADepth: integer = 64);
    destructor Destroy; override;

    procedure Start;
    procedure Stop;

    // Consumer. Blocks until a prepared sample is available, copies it into
    // the caller's volumes, then recycles the buffer. Safe for N concurrent
    // worker threads. If the prefetcher is stopped/draining it falls back to
    // building inline so a caller can never block forever.
    procedure GetPair(Input, Output: TNNetVolume);
  end;

implementation

type
  // Implementation-only: same-unit "private" access lets Execute reach the
  // owner's ProduceLoop. FLoader is typed as TThread in the interface.
  TKANLoaderThread = class(TThread)
  public
    Owner: TKANPrefetcher;
  protected
    procedure Execute; override;
  end;

procedure TKANLoaderThread.Execute;
begin
  Owner.ProduceLoop;
end;

{ TKANSamplePair }

constructor TKANSamplePair.Create;
begin
  inherited Create;
  Input := TNNetVolume.Create;
  Output := TNNetVolume.Create;
end;

destructor TKANSamplePair.Destroy;
begin
  Input.Free;
  Output.Free;
  inherited Destroy;
end;

{ TKANPrefetcher }

constructor TKANPrefetcher.Create(const ABuild: TKANBuildSampleProc;
  const ADepth: integer);
var
  I: integer;
begin
  inherited Create;
  FBuild := ABuild;
  FRunning := false;
  // Default (INFINITE) push/pop timeouts: producers/consumers block until
  // there is room / an item, or until DoShutDown releases them.
  FReady := TThreadedQueue<TKANSamplePair>.Create(ADepth);
  FFree  := TThreadedQueue<TKANSamplePair>.Create(ADepth);
  // Pre-fill the buffer pool so steady state allocates nothing per sample.
  for I := 0 to ADepth - 1 do
    FFree.PushItem(TKANSamplePair.Create);
end;

destructor TKANPrefetcher.Destroy;
var
  Pair: TKANSamplePair;
begin
  Stop;
  // Drain and free both pools. DoShutDown (in Stop) makes PopItem return a
  // non-signaled result once empty, so these loops terminate.
  while FReady.PopItem(Pair) = wrSignaled do Pair.Free;
  while FFree.PopItem(Pair)  = wrSignaled do Pair.Free;
  FReady.Free;
  FFree.Free;
  inherited Destroy;
end;

procedure TKANPrefetcher.Start;
var
  L: TKANLoaderThread;
begin
  if FRunning then exit;
  FRunning := true;
  L := TKANLoaderThread.Create(true);   // suspended
  L.Owner := Self;
  L.FreeOnTerminate := false;
  FLoader := L;
  L.Start;
end;

procedure TKANPrefetcher.Stop;
begin
  if not FRunning then exit;
  FRunning := false;
  // Release anyone blocked on a full ready-queue or an empty free-queue and
  // let ProduceLoop fall through. No thread can deadlock on shutdown.
  FReady.DoShutDown;
  FFree.DoShutDown;
  if Assigned(FLoader) then
  begin
    FLoader.WaitFor;
    FreeAndNil(FLoader);
  end;
end;

procedure TKANPrefetcher.ProduceLoop;
var
  Pair: TKANSamplePair;
begin
  while FRunning do
  begin
    if FFree.PopItem(Pair) <> wrSignaled then Break;   // shutting down
    FBuild(Pair.Input, Pair.Output);
    if FReady.PushItem(Pair) <> wrSignaled then
    begin
      Pair.Free;                                        // shutting down
      Break;
    end;
  end;
end;

procedure TKANPrefetcher.GetPair(Input, Output: TNNetVolume);
var
  Pair: TKANSamplePair;
begin
  if (not FRunning) or (FReady.PopItem(Pair) <> wrSignaled) then
  begin
    // Stopped or draining: build inline so the caller never blocks forever.
    FBuild(Input, Output);
    exit;
  end;
  Input.Copy(Pair.Input);
  Output.Copy(Pair.Output);
  Input.Tag := Pair.Input.Tag;
  Output.Tag := Pair.Output.Tag;
  FFree.PushItem(Pair);   // recycle the buffer
end;

end.
