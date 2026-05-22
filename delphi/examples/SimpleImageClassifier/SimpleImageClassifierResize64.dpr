program SimpleImageClassifierResize64;

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
  neuralab in '..\..\neural\neuralab.pas',
  neuralabfun in '..\..\neural\neuralabfun.pas',
  neuralbit in '..\..\neural\neuralbit.pas',
  neuralbyteprediction in '..\..\neural\neuralbyteprediction.pas',
  neuralcache in '..\..\neural\neuralcache.pas',
  neuralgeneric in '..\..\neural\neuralgeneric.pas',
  neuralsimd in '..\..\neural\neuralsimd.pas',
  neuralthread in '..\..\neural\neuralthread.pas';

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    procedure DoRun; override;
  end;

  procedure TTestCNNAlgo.DoRun;
  var
    NN: TNNet;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
  begin
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := TNNet.Create();
    NN.AddLayer([
      TNNetInput.Create(64, 64, 3),
      TNNetConvolutionLinear.Create(64, 5, 2, 1, 1),
      TNNetMaxPool.Create(4),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create({SkipBackpropDerivative=}1)
    ]);
    NN.DebugStructure();
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);
    ImgTrainingVolumes.ResizeImage(64, 64);
    ImgValidationVolumes.ResizeImage(64, 64);
    ImgTestVolumes.ResizeImage(64, 64);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifierResize64';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0;
    NeuralFit.MaxCropSize := 16;
    //NeuralFit.MaxThreadNum := 8;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}64, {epochs=}50);
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    Terminate;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='CIFAR-10 Classification Example';
  Application.Run;
  Application.Free;
end.
