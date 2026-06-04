unit checkpointcompanion;

(*
Checkpoint companion mechanism -- one class shared by training and inference so
the two sides cannot drift. Not KAN-specific: it companions any saved .nn.

A companion is a small human-readable sidecar written next to a saved .nn:

    context=84
    corpus_hash=<hex>
    nn_md5=<hex>

  * context     -- the context length the network was built at (positional
                   embeddings are sized to it; loading at the wrong value fails).
  * corpus_hash -- the dataset's chained HMAC-MD5 over its raw lines
                   (TKANTransformerDataset.CorpusHash).
  * nn_md5      -- MD5 of the .nn itself, so a consumer can confirm it is the
                   exact checkpoint this companion describes.

The path is <nnfile> + '.companion', pairing by basename.

TCheckpointCompanion encapsulates the whole mechanism -- path, format, the .nn
MD5, the comparisons, and the resolve/recover decision -- so the trainer (Write)
and the inference driver (Resolve) go through identical code:

  * Write   is idempotent: same .nn + same inputs => byte-identical file, so the
            trainer may call it on every save.
  * Resolve is read-only: same inputs => same decision, with no side effects.

The corpus hash and the recompute fallback are supplied by the caller (the
dataset), so this unit has no dataset dependency.
*)

interface

uses
  SysUtils, Classes, System.Hash;

type
  TCheckpointCompanion = class
  private
    FNNFileName: string;
    function CompanionPath: string;
  public
    constructor Create(const ANNFileName: string);

    // Trainer side. Write/refresh the companion describing FNNFileName as it is
    // on disk now, at AContextSize, for a corpus with hash ACorpusHash. The
    // .nn MD5 is computed here. Idempotent for fixed inputs + fixed .nn.
    procedure Write(const AContextSize: integer; const ACorpusHash: string);

    // Inference side. Resolve the context to build at:
    //   * companion present AND its corpus_hash + nn_md5 both match the current
    //     ones -> AContext := stored context; ARecovered := False.
    //   * otherwise (absent, unparseable, or content-mismatched) ->
    //     AContext := AFallbackContext; ARecovered := True.
    // Returns a one-line explanation of the decision (for printing). Read-only.
    function Resolve(const ACurrentCorpusHash: string;
      const AFallbackContext: integer;
      out AContext: integer; out ARecovered: boolean): string;
  end;

// Lowercase hex MD5 of a file, or '' if it does not exist. Shared helper.
function FileMD5(const AFileName: string): string;

implementation

function FileMD5(const AFileName: string): string;
begin
  if FileExists(AFileName) then
    Result := LowerCase(THashMD5.GetHashStringFromFile(AFileName))
  else
    Result := '';
end;

constructor TCheckpointCompanion.Create(const ANNFileName: string);
begin
  inherited Create;
  FNNFileName := ANNFileName;
end;

function TCheckpointCompanion.CompanionPath: string;
begin
  Result := FNNFileName + '.companion';
end;

procedure TCheckpointCompanion.Write(const AContextSize: integer;
  const ACorpusHash: string);
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.Add('context=' + IntToStr(AContextSize));
    L.Add('corpus_hash=' + ACorpusHash);
    L.Add('nn_md5=' + FileMD5(FNNFileName));
    L.SaveToFile(CompanionPath);
  finally
    L.Free;
  end;
end;

function TCheckpointCompanion.Resolve(const ACurrentCorpusHash: string;
  const AFallbackContext: integer;
  out AContext: integer; out ARecovered: boolean): string;
var
  L: TStringList;
  Path: string;
  StoredContext: integer;
  StoredCorpusHash, StoredNNMD5, CurNNMD5, CurCorpusHash: string;
begin
  Path := CompanionPath;

  if not FileExists(Path) then
  begin
    AContext := AFallbackContext;
    ARecovered := True;
    Result := Format('no companion for %s -- recovering: recomputed context = %d',
      [FNNFileName, AFallbackContext]);
    Exit;
  end;

  L := TStringList.Create;
  try
    L.LoadFromFile(Path);
    StoredContext := StrToIntDef(Trim(L.Values['context']), 0);
    StoredCorpusHash := LowerCase(Trim(L.Values['corpus_hash']));
    StoredNNMD5 := LowerCase(Trim(L.Values['nn_md5']));
  finally
    L.Free;
  end;

  CurNNMD5 := FileMD5(FNNFileName);
  CurCorpusHash := LowerCase(Trim(ACurrentCorpusHash));

  if (StoredContext > 0)
     and (CurCorpusHash = StoredCorpusHash)
     and (CurNNMD5 = StoredNNMD5) then
  begin
    AContext := StoredContext;
    ARecovered := False;
    Result := Format('companion valid (corpus + nn match) -- context = %d',
      [StoredContext]);
    Exit;
  end;

  // Not content-consistent -> recover, explaining why.
  AContext := AFallbackContext;
  ARecovered := True;
  if StoredContext <= 0 then
    Result := 'companion unparseable'
  else if CurCorpusHash <> StoredCorpusHash then
    Result := 'companion corpus-hash mismatch (dataset changed)'
  else
    Result := 'companion nn MD5 mismatch (not the checkpoint it describes)';
  Result := Result +
    Format(' -- recovering: recomputed context = %d', [AFallbackContext]);
end;

end.
