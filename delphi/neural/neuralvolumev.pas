(*
neuralvolumev
Copyright (C) 2017 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)

unit neuralvolumev;
// =====================================================================
// STANDALONE DELPHI PORT (modern Delphi 10.x - 12 Athens).
// Hand-converted from ../../neural/neuralvolumev.pas via ../_port_tool.py, which
// is Pascal-lexer-aware (directives in comments/strings untouched). FPC
// and the AVX defines are resolved undefined; FPC-only branches give
// way to the upstream Delphi branch. Canonical float/pointer types come
// from neuralsimd, re-exported through neuralvolume so call sites are
// unchanged. Review-verified only -- no Delphi toolchain was available.
// =====================================================================




interface

uses
  Classes, SysUtils, ExtCtrls, Graphics, neuralvolume,
  Winapi.Windows ;

/// saves a bitmap into a file from a handle HWND
procedure SaveHandleToBitmap(OutputFileName: string; hWnd: HWND);

/// Loads a volume into RGB TImage
procedure LoadVolumeIntoTImage(V:TNNetVolume; Image:TImage; color_encoding: integer = csEncodeRGB);

/// Loads a 3 layers RGB volume into RGB TImage
procedure LoadRGBVolumeIntoTImage(V:TNNetVolume; Image:TImage);

/// Loads a Picture into a Volume
procedure LoadPictureIntoVolume(LocalPicture: TPicture; Vol:TNNetVolume); {$IFDEF Release} inline; {$ENDIF}

/// Loads a Bitmat into a Volume
procedure LoadBitmapIntoVolume(LocalBitmap: Graphics.TBitmap; Vol:TNNetVolume);


procedure LoadImageFromFileIntoVolume(ImageFileName:string; V:TNNetVolume);


implementation


procedure SaveHandleToBitmap(OutputFileName: string; hWnd: HWND);

var
  MyBitmap: Graphics.TBitmap;
  MyDC    : HDC;
  pRect   : TRect;
  w,h     : integer;
begin
  MyDC := GetDC(hWnd);
  MyBitmap := Graphics.TBitmap.Create;
  try
    GetWindowRect(HWND,pRect);
    w  := pRect.Right - pRect.Left;
    h  := pRect.Bottom - pRect.Top;

    MyBitmap.Width := w;
    MyBitmap.Height:= h;

    BitBlt(MyBitmap.Canvas.Handle,
            0,
            0,
            MyBitmap.Width,
            MyBitmap.Height,
            MyDC,
            0,
            0,
            SRCCOPY) ;
    MyBitmap.SaveToFile(OutputFileName);
  finally
    ReleaseDC(hWnd, MyDC);
    FreeAndNil(MyBitmap);
  end;
end;


procedure LoadRGBVolumeIntoTImage(V:TNNetVolume; Image:TImage);
var
  I, J, MaxX, MaxY: integer;
begin
  MaxX := V.SizeX - 1;
  MaxY := V.SizeY - 1;

  for I := 0 to MaxX do
  begin
    for J := 0 to MaxY do
    begin
      Image.Canvas.Pixels[J, I] := RGB(V.AsByte[J,I,0], V.AsByte[J,I,1], V.AsByte[J,I,2]);
    end;
  end;
end;

procedure LoadPictureIntoVolume(LocalPicture: TPicture; Vol: TNNetVolume);
begin
  LoadBitmapIntoVolume(LocalPicture.Bitmap, Vol);
end;


procedure LoadImageFromFileIntoVolume(ImageFileName:string; V:TNNetVolume);
var
  LocalPicture: TPicture;
begin
  LocalPicture := TPicture.Create;
  LocalPicture.LoadFromFile( ImageFileName );
  LoadPictureIntoVolume(LocalPicture, V);
  LocalPicture.Free;
end;


procedure LoadBitmapIntoVolume(LocalBitmap: Graphics.TBitmap; Vol: TNNetVolume);
var
  CountX, CountY, MaxX, MaxY: integer;
  LocalCanvas: TCanvas;
  LocalColor: TColor;
  RawPos: integer;
begin
  LocalCanvas := LocalBitmap.Canvas;
  MaxX := LocalBitmap.Width - 1;
  MaxY := LocalBitmap.Height - 1;

  Vol.ReSize(MaxX + 1, MaxY + 1, 3);

  for CountX := 0 to MaxY do
  begin
    for CountY := 0 to MaxY do
    begin
      LocalColor := LocalCanvas.Pixels[CountX, CountY];
      RawPos := Vol.GetRawPos(CountX, CountY, 0);

      Vol.FData[RawPos]     := LocalColor          and $000000ff; // red
      Vol.FData[RawPos + 1] := (LocalColor shr 8)  and $000000ff; // green
      Vol.FData[RawPos + 2] := (LocalColor shr 16) and $000000ff; // blue
    end;
  end;
end;

procedure LoadVolumeIntoTImage(V:TNNetVolume; Image:TImage; color_encoding: integer = csEncodeRGB);
var
  I, J: integer;
  bG: byte;
  H,S,A,B,R,G: TNeuralFloat;
begin
  R := 0;
  G := 0;
  B := 0;

  if V.Depth = 1 then
  begin
    for I := 0 to V.SizeX - 1 do
    begin
      for J := 0 to V.SizeY - 1 do
      begin
        if color_encoding = csEncodeLAB then
        begin
          bG := RoundAsByte(V[J,I,0]*2.5);
        end
        else if ( (color_encoding = csEncodeHSL) or (color_encoding = csEncodeHSV) ) then
        begin
          bG := RoundAsByte(V[J,I,0]*255);
        end
        else
        begin
          // RGB and Gray
          bG := RoundAsByte(V[J,I,0]);
        end;
        Image.Canvas.Pixels[J, I] := RGB(bG, bG, bG);
      end;
    end;
  end
  else if V.Depth = 2 then
  begin
    for I := 0 to V.SizeX - 1 do
    begin
      for J := 0 to V.SizeY - 1 do
      begin
        if color_encoding = csEncodeLAB then
        begin
          A := V[J,I,0];
          B := V[J,I,1];
          lab2rgb(60, A, B, R, G, B);
        end
        else if color_encoding = csEncodeHSL then
        begin
          H := V[J,I,0];
          S := V[J,I,1];
          hsl2rgb(H, S, 0.6, R, G, B);
        end
        else if color_encoding = csEncodeHSV then
        begin
          H := V[J,I,0];
          S := V[J,I,1];
          hsv2rgb(H, S, 0.6, R, G, B);
        end
        else if color_encoding = csEncodeRGB then
        begin
          R := V[J,I,0];
          G := 0;
          B := V[J,I,1];
        end;
        Image.Canvas.Pixels[J, I] := RGB(RoundAsByte(R), RoundAsByte(G), RoundAsByte(B));
      end;
    end;
  end
  else if V.Depth > 2 then
  begin
    for I := 0 to V.SizeX - 1 do
    begin
      for J := 0 to V.SizeY - 1 do
      begin
        if color_encoding = csEncodeLAB then
        begin
          lab2rgb(V[J,I,0], V[J,I,1], V[J,I,2], R, G, B);
        end
        else if color_encoding = csEncodeHSL then
        begin
          hsl2rgb(V[J,I,0], V[J,I,1], V[J,I,2], R, G, B);
        end
        else if color_encoding = csEncodeHSV then
        begin
          hsv2rgb(V[J,I,0], V[J,I,1], V[J,I,2], R, G, B);
        end
        else if color_encoding = csEncodeRGB then
        begin
          R := V[J,I,0];
          G := V[J,I,1];
          B := V[J,I,2];
        end;
        //WriteLn(V[J,I,0]:10:5, ' ', V[J,I,1]:10:5, ' ', V[J,I,2]:10:5, ' - ', R:10:5, ' ', G:10:5, ' ', B:10:5);
        Image.Canvas.Pixels[J, I] := RGB(RoundAsByte(R), RoundAsByte(G), RoundAsByte(B));
      end;
    end;
  end;
end;

end.

