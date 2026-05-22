unit CustApp;

(*
  Minimal Delphi-compatible reimplementation of the FreePascal FCL `custapp`
  unit. Delphi has no CustApp unit and no TCustomApplication class -- its
  application class, Vcl.Forms.TApplication, is GUI/VCL-bound and is not a
  drop-in. The neural-api console examples (44 of them) are written as
  `class(TCustomApplication)` with an overridden DoRun, so this shim lets them
  compile and run under Delphi unchanged.

  Only the members the examples actually use are provided: Title, Run/DoRun,
  Terminate, HasOption, GetOptionValue and StopOnException. Run is one-shot
  (calls DoRun once); every example's DoRun does its whole job in a single
  call and then calls Terminate, matching this behaviour.
*)

interface

uses
  System.Classes, System.SysUtils;

type
  TCustomApplication = class(TComponent)
  private
    FTitle: string;
    FTerminated: Boolean;
    FStopOnException: Boolean;
  protected
    procedure DoRun; virtual;
  public
    constructor Create(AOwner: TComponent); override;
    procedure Run;
    procedure Terminate;
    function HasOption(const C: Char; const S: string): Boolean;
    function GetOptionValue(const C: Char; const S: string): string;
    property Title: string read FTitle write FTitle;
    property Terminated: Boolean read FTerminated;
    property StopOnException: Boolean read FStopOnException write FStopOnException;
  end;

implementation

constructor TCustomApplication.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FTerminated := False;
  FStopOnException := False;
end;

procedure TCustomApplication.DoRun;
begin
end;

procedure TCustomApplication.Run;
begin
  try
    DoRun;
  except
    on E: Exception do
    begin
      Writeln(ErrOutput, E.ClassName, ': ', E.Message);
      if FStopOnException then
        raise;
    end;
  end;
end;

procedure TCustomApplication.Terminate;
begin
  FTerminated := True;
end;

function TCustomApplication.HasOption(const C: Char; const S: string): Boolean;
var
  I: Integer;
  P, ShortPfx, LongPfx: string;
begin
  Result := False;
  ShortPfx := '-' + C;
  LongPfx := '--' + S;
  for I := 1 to ParamCount do
  begin
    P := ParamStr(I);
    if (P = ShortPfx) or (P = LongPfx) or
       (Copy(P, 1, Length(ShortPfx)) = ShortPfx) or
       (Copy(P, 1, Length(LongPfx) + 1) = LongPfx + '=') then
      Exit(True);
  end;
end;

function TCustomApplication.GetOptionValue(const C: Char; const S: string): string;
var
  I: Integer;
  P, ShortPfx, LongPfx: string;
begin
  Result := '';
  ShortPfx := '-' + C;
  LongPfx := '--' + S;
  for I := 1 to ParamCount do
  begin
    P := ParamStr(I);
    if (P = ShortPfx) or (P = LongPfx) then
    begin
      if I < ParamCount then
        Result := ParamStr(I + 1);
      Exit;
    end
    else if Copy(P, 1, Length(LongPfx) + 1) = LongPfx + '=' then
      Exit(Copy(P, Length(LongPfx) + 2, MaxInt))
    else if Copy(P, 1, Length(ShortPfx)) = ShortPfx then
    begin
      Result := Copy(P, Length(ShortPfx) + 1, MaxInt);
      if (Result <> '') and (Result[1] = '=') then
        Delete(Result, 1, 1);
      Exit;
    end;
  end;
end;

end.
