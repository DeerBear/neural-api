# Memory-Mapped Text Dataset Loader

`TNNetMappedTextDataset` (unit `neuralmmapdataset`) is a line-oriented text
corpus loader for `TNeuralDataLoadingFit`. It maps the dataset file **once**,
read-only, and lets every worker thread read sample text straight from that one
shared mapping — no per-thread copy of the corpus, no eager `TStringList` load.

It is self-contained: it depends only on `neuralvolume` and `neuralnetwork`, so
it drops into any `FitLoading`-based example unchanged.

---

## Example

Swapping a `TStringList`-backed loader for the memory-mapped one is a four-line
change. Build your network as usual, then:

```pascal
uses
  neuralmmapdataset;

var
  Mmap: TNNetMappedTextDataset;
...
  // 1. Create with the minimum kept-line length (3 = the SimpleNLP default).
  Mmap := TNNetMappedTextDataset.Create({MinSampleSize=}3);

  // 2. Map + index the file once. Prints progress while it scans.
  Mmap.LoadDataset('datasets/tinystories.txt');

  // 3. Bind the network so samples can be sized from its first/last layers.
  //    Call this AFTER the network is built.
  Mmap.BindNetwork(NN);

  // 4. Hand the getters to FitLoading. Signatures already match.
  NFit.FitLoading(
    NN,
    Mmap.Count,            // training pairs = number of kept lines
    ValidationCount,
    0,                     // test pairs
    BatchSize,
    @Mmap.GetTrainingPair, // random sampling
    @Mmap.GetValidationPair, // deterministic in Idx
    @Mmap.GetValidationPair);
...
  Mmap.Free;
```

### Before → after

The stock `examples/SimpleNLP` loader holds the whole corpus in a `TStringList`
and reads samples out of it:

```pascal
FDataset := TStringList.Create();
LoadDataset();                 // ReadLn every line into FDataset
...
SampleId := Random(FDatasetSize);
... copy(FDataset[SampleId], 1, SampleCutPosition) ...
```

The mapped loader keeps the same sample semantics but drops the in-memory
corpus entirely:

```pascal
Mmap := TNNetMappedTextDataset.Create(3);
Mmap.LoadDataset('datasets/tinystories.txt'); // map once, index offsets
Mmap.BindNetwork(NN);
... // GetTrainingPair / GetValidationPair read from the shared mapping
```

The produced `(input, output)` pairs are identical in encoding (see
[Encoding parity](#encoding-parity)), so accuracy/loss are unchanged; only the
memory and I/O behaviour differ.

---

## Technical decisions

### Load once, work downstream

A `TStringList` loader holds one full heap copy of the corpus, and under
data-parallel training the read pattern fans out across worker threads. The
mapped loader replaces that with a single read-only mapping backed by the OS
page cache:

- **One copy, not N.** Resident memory is the page cache for the touched pages,
  not a heap-resident `TStringList`. On a multi-GB corpus this is the
  difference between "fits" and "doesn't" on a small box.
- **Demand paging overlaps I/O with compute.** Pages are faulted in only when a
  sample touches them. When worker threads outnumber cores, a thread stalled on
  a page fault yields the core to a thread that has its data resident — the OS
  hides I/O latency behind compute. This is the property that makes
  oversubscribed worker counts worthwhile here.
- **The loading is shared; the per-sample work is downstream of it.** Indexing
  happens once, up front; every thread's encoding work is independent and reads
  from the same bytes.

### Independence

The unit pulls in only the two core units it genuinely needs —
`neuralvolume` (the sample volumes) and `neuralnetwork` (to size those volumes
from the bound network's first/last layers). It references no optimiser, no
attention, and no other optional machinery, so it works regardless of what the
rest of the network is made of and can be contributed and reviewed on its own.

### Cross-platform mapping

Memory mapping is the one platform-specific part, isolated behind `{$IFDEF}`:

| Platform | Open | Map | Unmap |
|---|---|---|---|
| Unix (Linux/macOS/BSD) | `FpOpen(O_RDONLY)` | `Fpmmap(PROT_READ, MAP_PRIVATE)` | `Fpmunmap` |
| Windows | `CreateFile(GENERIC_READ)` | `CreateFileMapping(PAGE_READONLY)` + `MapViewOfFile(FILE_MAP_READ)` | `UnmapViewOfFile` |

Everything above the mapping — the one-pass line index, line extraction, and
sample building — is platform-independent and shared.

### Drop-in getters

`GetTrainingPair` and `GetValidationPair` use the
`(Idx, ThreadId: integer; pInput, pOutput: TNNetVolume)` signature that
`TNeuralDataLoadingFit` expects, so they can be passed to `FitLoading` directly
with no wrapper. `BindNetwork` must be called first so the getters can resize
`pInput`/`pOutput` to the network's first/last layer shapes.

- **Training** samples a random line each call (`Idx` is ignored), matching the
  stock getter's behaviour.
- **Validation** is deterministic in `Idx` (line and cut position are both
  derived from it), so validation passes are reproducible run to run.

### Encoding parity

Samples reproduce the SimpleNLP char-level convention exactly, so the loader is
a true drop-in rather than a behavioural variant:

- Each kept line is **lowercased** and a **sentinel byte `#1`** is appended.
- The **input** is `OneHotEncodingReversed` of a prefix of the line.
- The **target** is the next character as a softmax class
  (`SetClassForSoftMax`), with `pOutput.Tag` carrying the integer token.
- Lines shorter than `MinSampleSize` are skipped during indexing.

### Indexing and safety

- The index is built in a **single forward pass**: split on `LF`, trim a
  trailing `CR` (matching `ReadLn` semantics), keep each line's byte offset and
  length. Progress is printed every ~256 MB so a multi-GB scan is not a silent
  gap before the first training log.
- Line-offset arithmetic goes through `PtrUInt`/`Int64`, so the index is safe
  past 2 GB on 64-bit builds.
- The mapping is **never mutated**. Lines are copied out (and lowercased +
  sentinel-appended) only when a sample is built; the mapped pages stay
  read-only.

### Optional knob: `MaxPredictCharPos`

Mirrors the stock example's clamp on the **training** prefix length. `0`
(default) disables it. Validation is never clamped.

---

## Notes and limitations

- **Training randomness.** `GetTrainingPair` draws samples with the global
  `Random`, exactly as the stock `TStringList`-backed getter does. This is not
  thread-safe in the strict sense across worker threads, but it matches the
  existing example's behaviour; sampling bias from the occasional race is
  negligible for shuffled training.
- **Validate on a real build.** If contributing or porting, do a real
  `fpc`/`lazbuild` pass. The platform symbols to confirm are, on Unix,
  `FpOpen` / `FpLseek` / `Fpmmap` / `Fpmunmap` / `FpClose` / `PROT_READ` /
  `MAP_PRIVATE` / `O_RDONLY` / `SEEK_END` / `SEEK_SET` / `cint` / `fpgeterrno`
  (all from `BaseUnix`); on Windows, the `CreateFileMapping` / `MapViewOfFile`
  set from `Windows`.
