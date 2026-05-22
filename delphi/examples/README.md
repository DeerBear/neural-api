# Delphi examples

Delphi console versions of the neural-api example programs. They are the
Lazarus/FPC examples from [`examples/`](../../examples) converted to Delphi and
laid out here to mirror that directory, alongside the standalone Delphi port of
the library in [`delphi/neural/`](../neural).

## Requirements

- Delphi 10.x – 12 (Athens), Win32 or Win64.
- The example datasets (CIFAR-10, MNIST, Fashion-MNIST, plant-leaf images,
  etc.). As with the Lazarus examples, datasets are loaded by relative path, so
  run each program from a working directory that contains the data it expects.

## Building and running

Each project is a single self-contained `.dpr`. **Just open the `.dpr` in the
Delphi IDE** (or compile it with `dcc32`/`dcc64`) — Delphi creates the `.dproj`
on first open. No library path or project configuration is needed: every unit
in each `.dpr`'s `uses` clause carries an explicit relative path
(`neuralnetwork in '..\..\neural\neuralnetwork.pas'`, etc.) pointing at
`delphi/neural/` and at the `CustApp` shim, so the library units are picked up
automatically.

## The `CustApp` shim

`CustApp.pas` in this directory is a minimal Delphi-compatible reimplementation
of FreePascal's `custapp` unit. Delphi has no `TCustomApplication` class (its
application class, `Vcl.Forms.TApplication`, is GUI-bound), and most examples
are written as `class(TCustomApplication)` with an overridden `DoRun`. The shim
provides just the members the examples use — `Title`, `Run`/`DoRun`,
`Terminate`, `HasOption`, `GetOptionValue`, `StopOnException` — so the example
sources compile under Delphi unchanged. `{$APPTYPE CONSOLE}` only selects the
console binary subsystem; it does not supply this class.

## Known limitations

Three groups compile only partially and are kept here on a best-effort basis:

- **`SuperResolution/`** (`SuperResolution`, `SuperResolutionTrain`,
  `Cifar10ImageClassifierSuperResolution`) use FreePascal's `FPImage` /
  `FPWrite*` image-I/O units, which have no Delphi equivalent. The image I/O
  needs reworking (e.g. to `Vcl.Imaging.*`) before these build.
- **`SimpleImageClassifierGPU/`** uses OpenCL via `neuralopencl`; it also needs
  the [`delphi/opencl`](../opencl) (DelphiCL) submodule.
- **`ResNet/server/ResNetServer`** is a web-server example built on
  FreePascal-only units (`fphttpapp`, `httproute`, LCL `Interfaces`/`Graphics`);
  it does not port directly to Delphi.

## How these were generated

The `.dpr` files are produced from the Lazarus examples by
[`delphi/_port_examples.py`](../_port_examples.py), a companion to the library
porter `_port_tool.py`. It reuses the same reviewed, comment-aware Pascal lexer
to resolve `{$IFDEF FPC/AVX/UNIX/UseCThreads}` branches, strips `{$mode}`
directives, adds `{$APPTYPE CONSOLE}`, expands compound assignments, and
rewrites each `uses` clause with explicit unit paths. Re-running it regenerates
this tree deterministically.

## Not included

The four LCL-GUI examples (`VisualGAN`, `VisualAutoencoder`, `GradientAscent`,
`SuperResolution/SuperResolutionApp`) are not converted here — porting their
Lazarus forms to Delphi/VCL is tracked separately.
