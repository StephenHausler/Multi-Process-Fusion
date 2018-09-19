%--------------------------------------------------------------------------
%   Title: Multi-Process Fusion
%   Author: Stephen Hausler
%   
%   Open Source Code, requires MATLAB with Neural Network Toolbox
%   Refer to LICENSES.txt for license to this source code and all 3rd party
%   licences.
%--------------------------------------------------------------------------

clear variables
% clear global        
% %global variables:
% global Nordland_tunnel_skip
% global id2Vid

%--------------------------------------------------------------------------
%START OF ADJUSTABLE SETTINGS
%Dataset load:
%is it images or video:
Video_option = 0;  % 0 = images, 1 = video
Nordland_tunnel_skip = 0;   %only set to 1 if using Nordland video and want
%to skip all the tunnels (skips are hardcoded in DatabaseLoad.m).

%Reference Database File Address (provide full path to folder containing images or video):
Ref_folder = 'D:\Windows\St_Lucia_Dataset\0845_15FPS\Frames';
Ref_file_type = '.jpeg';     
Imstart_R = 0;  %start dataset after this many frames or seconds for video.
finalImage_R = 3945; %For St Lucia dataset, this is where the first loop ends

%Query Database File Address (provide full path to folder containing images or video):
Query_folder = 'D:\Windows\St_Lucia_Dataset\1545_15FPS\Frames';
Query_file_type = '.jpeg';
Imstart_Q = 0;  %start dataset after this many frames or seconds for video.
finalImage_Q = 4000;

Frame_skip = 1;     %Duel use variable: for images, this means use every ith frame
%for videos, this means extract out of video at this FPS.

%Ground truth load:
%Load a ground truth correspondance matrix.
GT_file = load('D:\Windows\St_Lucia_Dataset\StLucia_GPSMatrix.mat');

%Neural Network load:
% datafile = './HybridNet/HybridNet.caffemodel';
% protofile = './HybridNet/deploy.prototxt';
datafile = 'D:\Windows\MATLAB\Caffe_model_vgg16/vgg16_places365.caffemodel';
protofile = 'D:\Windows\MATLAB\Caffe_model_vgg16/deploy_vgg16_places365.prototxt';
net = importCaffeNetwork(protofile,datafile);

%Algorithm adjustable settings:
%59 thresholds: this generates the precision/recall curves.
thresh = [0.0001 0.001 0.005 0.01 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16...
    0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4 0.42 0.44...
    0.46 0.48 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.1 1.2...
    1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.5 3 3.5 4 4.5 5 6 8 10 100]; 
maxSeqLength = 20;      %maximum sequence length 
minSeqLength = 5;       %miniumum sequence length 
obsThresh = 0.5;        %threshold for flooring observations to epsilon
epsilon = 0.001;        %floor value
minVelocity = 0;        %minimum assumed velocity of vehicle in number of frames.
maxVelocity = 5;        %maximum assumed velocity of vehicle in number of frames.
Qt = 0.1;               %Quality rate-of-change threshold 
%(the minimum ROC to trigger a detection in a change of environment novelty)
Rwindow = 20;           %Change this to reflect the approximate distance 
%between frames and the localisation accuracy required.
%In the paper, we used a Rwindow of 30 for St Lucia and Campus, 20 for
%RobotCar and 10 for Nordland.
%However, post submission we discovered that a Rwindow of 20 is better for
%St Lucia since we achieve 63% recall at 100% precision rather than 47%
%recall at 100% precision.

%Boolean Flags setting algorithm options
Normalise = 1;          % True: Perform normalisation on CNN feature vectors
qROC_Smooth = 0;        % True: Smooth quality scores over sequence using a moving average
% False: (default) Use raw quality scores for R.O.C. calculation
% Note: As a rule-of-thumb, half Qt for smoothing option.
% Published results use false (default), however the users of this code are
% welcome to use either option to see what works best for your specific
% application.

%put the query traverse specific settings into a struct for improved
%readability
algSettings = struct('thresh',thresh,'maxSeqLength',maxSeqLength,'minSeqLength',...
    minSeqLength,'obsThresh',obsThresh,'minVelocity',minVelocity,'maxVelocity',...
    maxVelocity,'Qt',Qt,'Rwindow',Rwindow,'qROC_Smooth',qROC_Smooth,'epsilon',epsilon);

%TODO: allow for different types of user-provided image processing code
%files.
%Image processing method adjustable settings:
Initial_crop = [20 60 0 0];  %crop amount in pixels top, bottom, left, right
Initial_crop = Initial_crop + 1;    %only needed because MATLAB starts arrays from index 1.

SAD_resolution = [64 32];       %width by height
SAD_patchSize = 8;

HOG_resolution = [640 320];     %width by height
HOG_cellSize = [32 32];         

actLayer = 24;      %default for HybridNet is Conv-5 ReLu layer
%Need to adjust this for the neural network you use.

%END OF ADJUSTABLE SETTINGS
%--------------------------------------------------------------------------

%Run reference traverse
[Template_array1,Template_array2,Template_array3,Template_array4,...
    totalImagesR,Template_count,Template_plot] = ...
    DatabaseLoad(Video_option,Ref_folder,Ref_file_type,Imstart_R,...
    Frame_skip,net,actLayer,SAD_resolution,SAD_patchSize,HOG_resolution,...
    HOG_cellSize,Initial_crop,Normalise,finalImage_R);

%Now run query traverse...
[precision,recall,truePositive,falsePositive,worstIDCounter] = Multi_Process_Fusion_Run(...
    Video_option,Ref_folder,Ref_file_type,Query_folder,Query_file_type,...
    Imstart_Q,Imstart_R,Frame_skip,net,actLayer,SAD_resolution,SAD_patchSize,...
    HOG_resolution,HOG_cellSize,Initial_crop,Normalise,Template_array1,...
    Template_array2,Template_array3,Template_array4,GT_file,algSettings,...
    finalImage_Q,totalImagesR,Template_count,Template_plot);

%Now display results:

precision
recall

%Plot the filter proportions:
figure
labels = {'CNN','CNND','HOG','SAD'};
pie(worstIDCounter,labels);














