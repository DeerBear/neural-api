unit kanweightremix;

(*
Surgical weight loading for neural-api networks.

Provides selective checkpoint loading: load weights from a saved file
into only the specified layer indices, leaving every other layer
untouched. Use case: recover from a collapsed training run by rolling
back the broken layers (typically Q/K projections in attention blocks)
to a pre-collapse checkpoint while keeping the more-trained weights
everywhere else.

File format mirrors TNNet.LoadDataFromFile:

  header > data

(split on '>'), where the data section is a sequence of per-layer
weight strings separated by '!'. This helper parses the same format
but applies the layer-N data only when N is in the caller-supplied
target index set.
*)

interface

uses
  Classes, SysUtils, neuralnetwork, neuralvolume;

type
  TLayerIndexSet = array of integer;

// Load weights from Filename, but only into the layers whose indices
// appear in TargetIndices. The file must match pNN's structure in
// layer count; mismatched layer counts raise an exception.
procedure LoadLayersFromFile(pNN: TNNet; const Filename: string;
  const TargetIndices: array of integer);

implementation

function IsTarget(Idx: integer;
  const TargetIndices: array of integer): boolean;
var
  I: integer;
begin
  for I := Low(TargetIndices) to High(TargetIndices) do
    if TargetIndices[I] = Idx then
    begin
      Result := True;
      exit;
    end;
  Result := False;
end;

procedure LoadLayersFromFile(pNN: TNNet; const Filename: string;
  const TargetIndices: array of integer);
var
  Outer, Sec: TNNetStringList;
  DataStr: string;
  Cnt, Loaded: integer;
begin
  if Length(TargetIndices) = 0 then exit;

  Outer := CreateTokenizedStringList('>');
  try
    Outer.LoadFromFile(Filename);
    if Outer.Count <> 2 then
      raise Exception.CreateFmt(
        'LoadLayersFromFile: bad file format in %s ' +
        '(expected 2 sections separated by ''>'', got %d)',
        [Filename, Outer.Count]);
    DataStr := Outer[1];

    Sec := CreateTokenizedStringList(DataStr, '!');
    try
      if Sec.Count <> pNN.CountLayers then
        raise Exception.CreateFmt(
          'LoadLayersFromFile: layer count mismatch in %s ' +
          '(file has %d layers, network has %d)',
          [Filename, Sec.Count, pNN.CountLayers]);

      Loaded := 0;
      for Cnt := 0 to Sec.Count - 1 do
      begin
        if IsTarget(Cnt, TargetIndices) then
        begin
          pNN.Layers[Cnt].LoadDataFromString(Sec[Cnt]);
          Inc(Loaded);
        end;
      end;
      WriteLn(Format(
        'LoadLayersFromFile: replaced %d layer(s) from %s (target set size: %d)',
        [Loaded, Filename, Length(TargetIndices)]));
    finally
      Sec.Free;
    end;
  finally
    Outer.Free;
  end;
end;

end.
