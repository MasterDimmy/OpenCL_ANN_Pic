# OpenCL_ANN_Pic
Simple C++ Classificator of pictures, used OpenCV Artificial Neural Network (ANN)

# Files
example\    - usage example 
-test_images - folder with images to learn
-test.jpg - image to classify
-config.ini - config in format, described below
-opencv_world310.dll - opencv dll for windows
-save.ann - saved ANN
-save.tp - saved class names
-src.exe - compiled app

src\ 
-src.cpp - source file, comments are written in Russian

# Config
config.ini:
local_path_to_images_in_subfoders
path_to_image_for_classification

# Build 
For MSVS 2008 for Windows 7 x64
1. place OpenCV_3.2.0 to C:\opencv
2. run cmd.exe
3. in cmd.exe type :>\  "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\amd64\vcvarsamd64.bat"
4. goto folder with "src.cpp" in cmd.exe via "cd" command
5. cl.exe /F 1024000000 /I C:\opencv\build\include /w /wd4530  /wd4244 src.cpp   opencv_world310.lib  /link /libpath:C:\opencv\build\x64\vc12\lib

you got runable src.exe

# Algorithm
src.exe 
1. reads content of config.ini
2. reads database files (save * ) if they exists to ANN, otherwise creates ANN and study it
3. calculate precision of test.jpg
4. output result

# Demo run of example
\>src.exe

image database folder = test_images

test image = test.jpg

try to load saved ANN: save.ann ...

loaded saved ANN

loaded 2 types

loading test image

calculating prediction

test image  [ 0.89, -0.90 ]

best type num: 0

test image type: A
