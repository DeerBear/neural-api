program SelfTest;

{$APPTYPE CONSOLE}
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)


uses
  Classes,
  SysUtils,
  CustApp in '..\CustApp.pas',
  neuralnetwork in '..\..\neural\neuralnetwork.pas',
  neuralvolume in '..\..\neural\neuralvolume.pas',
  Math,
  neuraldatasets in '..\..\neural\neuraldatasets.pas',
  neuralfit in '..\..\neural\neuralfit.pas',
  neuralthread in '..\..\neural\neuralthread.pas',
  neuralab in '..\..\neural\neuralab.pas',
  neuralabfun in '..\..\neural\neuralabfun.pas',
  neuralbit in '..\..\neural\neuralbit.pas',
  neuralbyteprediction in '..\..\neural\neuralbyteprediction.pas',
  neuralcache in '..\..\neural\neuralcache.pas',
  neuralgeneric in '..\..\neural\neuralgeneric.pas',
  neuralsimd in '..\..\neural\neuralsimd.pas';

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    procedure DoRun; override;
  end;

  procedure TTestCNNAlgo.DoRun;
  begin
    WriteLn('Testing Volumes API ...');
    TestTNNetVolume();
    TestKMeans();

    WriteLn('Testing Convolutional API ...');
    TestConvolutionAPI();

    WriteLn('Press ENTER to quit.');
    ReadLn();
    Terminate;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='Neural API self-test';
  Application.Run;
  Application.Free;
end.
