%--------------------------------------------------------------------------
%   Title: Multi-Process Fusion
%   Author: Stephen Hausler
%   
%   Open Source Code, requires MATLAB with Neural Network Toolbox
%   Refer to LICENSES.txt for license to this source code and 3RD_PARTY_
%   LICENSES for all 3rd party licences.
%--------------------------------------------------------------------------

clear variables 

global PlotOption   %this variables sets if plots are generated
PlotOption = 1;

%Dataset Selection:
    Lucia = 0;
    Oxford = 1;
    Nordland = 1;
    Campus = 1;
%Set each to 1 to run each dataset. Please note that each dataset has its own individual settings.

%--------------------------------------------------------------------------
if Lucia == 1
%START OF ADJUSTABLE SETTINGS
%Dataset load:
NordlandGT = 0;           % only used for the Nordland dataset

%Reference Database File Address (provide full path to folder containing images or video):
Ref_folder = 'D:\Windows\St_Lucia_Dataset\0845_15FPS\Frames';   
Ref_file_type = '.jpeg';     
Imstart_R = 0;  %start dataset after this many frames or seconds for video.
finalImage_R = 3945; %For St Lucia dataset, this is where the first loop ends
%finalImage_R = 394;

%Query Database File Address (provide full path to folder containing images or video):
Query_folder = 'D:\Windows\St_Lucia_Dataset\1545_15FPS\Frames';
Query_file_type = '.jpeg';
Imstart_Q = 0;  %start dataset after this many frames or seconds for video.
finalImage_Q = 4000;
%finalImage_Q = 400;

Frame_skip = 1;     %Duel use variable: for images, this means use every ith frame
%for videos, this means extract frames out of video at this FPS.

%Ground truth load:
%Load a ground truth correspondance matrix.
GT_file = load('D:\Windows\St_Lucia_Dataset\StLucia_GPSMatrix_20m.mat');
%GT_file = load('D:\Windows\oxford-data\OxfordRobotCar_GPSMatrix_30m.mat');

%Neural Network load:
datafile = 'D:\MATLAB\MPF_RevisePaper\HybridNet\HybridNet.caffemodel';
protofile = 'D:\MATLAB\MPF_RevisePaper\HybridNet\deploy.prototxt';
% datafile = 'D:\Windows\MATLAB\Caffe_model_vgg16/vgg16_places365.caffemodel';
% protofile = 'D:\Windows\MATLAB\Caffe_model_vgg16/deploy_vgg16_places365.prototxt';
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
plotThresh = 0.4;       %Quality threshold for template plot graph
%(the minimum ROC to trigger a detection in a change of environment novelty)
Rwindow = 20;           %Change this to reflect the approximate distance 
%between frames and the localisation accuracy required.
%Default value (which has been experimentally tested as being a suitable default value) is 20 frames.

%Boolean Flags setting algorithm options
Normalise = 1;          % True: Perform normalisation on CNN feature vectors
qROC_Smooth = 0;        % True: Smooth quality scores over sequence using a moving average
% False: (default) Use raw quality scores for R.O.C. calculation
% Note: As a rule-of-thumb, half Qt for smoothing option.
% Published results use false (default), however the users of this code are
% welcome to use either option to see what works best for your specific
% application.

algSettings = struct('thresh',thresh,'maxSeqLength',maxSeqLength,'minSeqLength',...
    minSeqLength,'obsThresh',obsThresh,'minVelocity',minVelocity,'maxVelocity',...
    maxVelocity,'Qt',Qt,'Rwindow',Rwindow,'qROC_Smooth',qROC_Smooth,...
    'epsilon',epsilon,'plotThresh',plotThresh);

%Image processing method adjustable settings:
Initial_crop = [0 60 0 0];  %crop amount in pixels top, bottom, left, right
Initial_crop = Initial_crop + 1;    %only needed because MATLAB starts arrays from index 1.

SAD_resolution = [64 32];       %width by height
SAD_patchSize = 8;

HOG_resolution = [640 320];     %width by height
HOG_cellSize = [32 32];         

CNN_resolution = [227 227];     %224 by 224 for vgg-16 and 227 by 227 for HybridNet

%actLayer = 24; %for VGG-16 (note: VGG-16 will run slower than HybridNet)     
actLayer = 15; %default for HybridNet is Conv-5 ReLu layer
%Need to adjust this for the neural network you use.

%END OF ADJUSTABLE SETTINGS
%--------------------------------------------------------------------------

%Run reference traverse
[Template_array1,Template_array2,Template_array3,Template_array4,...
    totalImagesR,Template_count,Template_plot] = ...
    DatabaseLoad(NordlandGT,Ref_folder,Ref_file_type,Imstart_R,...
    Frame_skip,net,actLayer,SAD_resolution,SAD_patchSize,HOG_resolution,...
    HOG_cellSize,CNN_resolution,Initial_crop,Normalise,finalImage_R);

Mem = whos;

%Now run query traverse...
[precision,recall,truePositive,falsePositive,worstIDCounter,AverageComputeTime,TotalComputeTime] = Multi_Process_Fusion_Run(...
    NordlandGT,Ref_folder,Ref_file_type,Query_folder,Query_file_type,...
    Imstart_Q,Imstart_R,Frame_skip,net,actLayer,SAD_resolution,SAD_patchSize,...
    HOG_resolution,HOG_cellSize,CNN_resolution,Initial_crop,Normalise,Template_array1,...
    Template_array2,Template_array3,Template_array4,GT_file,algSettings,...
    finalImage_Q,totalImagesR,Template_count,Template_plot);

save('MPF_Lucia.mat','precision','recall','truePositive',...
    'falsePositive','worstIDCounter','AverageComputeTime','TotalComputeTime','Mem');

end 
%-------Next run Oxford RobotCar-------------------------------------------
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%--------------------------------------------------------------------------
if Oxford == 1
    
%clear variables      

%--------------------------------------------------------------------------
%START OF ADJUSTABLE SETTINGS
%Dataset load:
NordlandGT = 0;           % only used for the Nordland dataset

%Reference Database File Address (provide full path to folder containing images or video):
Ref_folder = 'D:\Windows\oxford-data\2014-12-09-13-21-02\stereo\left_rect';   
Ref_file_type = '.png';     
Imstart_R = 0;  %start dataset after this many frames or seconds for video.
finalImage_R = 5602; 
%finalImage_R = 598;

%Query Database File Address (provide full path to folder containing images or video):
Query_folder = 'D:\Windows\oxford-data\2014-12-10-18-10-50\stereo\left_rect';
Query_file_type = '.png';
Imstart_Q = 0;  %start dataset after this many frames or seconds for video.
finalImage_Q = 6074;
%finalImage_Q = 782;

Frame_skip = 3;     %Duel use variable: for images, this means use every ith frame
%for videos, this means extract frames out of video at this FPS.

%Ground truth load:
%Load a ground truth correspondance matrix.
GT_file = load('D:\Windows\oxford-data\OxfordRobotCar_GPSMatrix_30m.mat');

%Neural Network load:
datafile = 'D:\MATLAB\MPF_RevisePaper\HybridNet\HybridNet.caffemodel';
protofile = 'D:\MATLAB\MPF_RevisePaper\HybridNet\deploy.prototxt';
% datafile = 'D:\Windows\MATLAB\Caffe_model_vgg16/vgg16_places365.caffemodel';
% protofile = 'D:\Windows\MATLAB\Caffe_model_vgg16/deploy_vgg16_places365.prototxt';
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
plotThresh = 0.4;       %Quality threshold for template plot graph
%(the minimum ROC to trigger a detection in a change of environment novelty)
Rwindow = 20;           %Change this to reflect the approximate distance 
%between frames and the localisation accuracy required.
%Default value (which has been experimentally tested as being a suitable default value) is 20 frames.

%Boolean Flags setting algorithm options
Normalise = 1;          % True: Perform normalisation on CNN feature vectors
qROC_Smooth = 0;        % True: Smooth quality scores over sequence using a moving average
% False: (default) Use raw quality scores for R.O.C. calculation
% Note: As a rule-of-thumb, half Qt for smoothing option.
% Published results use false (default), however the users of this code are
% welcome to use either option to see what works best for your specific
% application.

algSettings = struct('thresh',thresh,'maxSeqLength',maxSeqLength,'minSeqLength',...
    minSeqLength,'obsThresh',obsThresh,'minVelocity',minVelocity,'maxVelocity',...
    maxVelocity,'Qt',Qt,'Rwindow',Rwindow,'qROC_Smooth',qROC_Smooth,...
    'epsilon',epsilon,'plotThresh',plotThresh);

%Image processing method adjustable settings:
Initial_crop = [0 140 0 0];  %crop amount in pixels top, bottom, left, right
Initial_crop = Initial_crop + 1;    %only needed because MATLAB starts arrays from index 1.

SAD_resolution = [64 32];       %width by height
SAD_patchSize = 8;

HOG_resolution = [640 320];     %width by height
HOG_cellSize = [32 32];         

CNN_resolution = [227 227];     %224 by 224 for vgg-16 and 227 by 227 for HybridNet

%actLayer = 24;  %for VGG-16 (note: VGG-16 will run slower than HybridNet)         
actLayer = 15; %default for HybridNet is Conv-5 ReLu layer
%Need to adjust this for the neural network you use.

%END OF ADJUSTABLE SETTINGS
%--------------------------------------------------------------------------

%Run reference traverse
[Template_array1,Template_array2,Template_array3,Template_array4,...
    totalImagesR,Template_count,Template_plot] = ...
    DatabaseLoad(NordlandGT,Ref_folder,Ref_file_type,Imstart_R,...
    Frame_skip,net,actLayer,SAD_resolution,SAD_patchSize,HOG_resolution,...
    HOG_cellSize,CNN_resolution,Initial_crop,Normalise,finalImage_R);

Mem = whos;

%Now run query traverse...
[precision,recall,truePositive,falsePositive,worstIDCounter,AverageComputeTime,TotalComputeTime] = Multi_Process_Fusion_Run(...
    NordlandGT,Ref_folder,Ref_file_type,Query_folder,Query_file_type,...
    Imstart_Q,Imstart_R,Frame_skip,net,actLayer,SAD_resolution,SAD_patchSize,...
    HOG_resolution,HOG_cellSize,CNN_resolution,Initial_crop,Normalise,Template_array1,...
    Template_array2,Template_array3,Template_array4,GT_file,algSettings,...
    finalImage_Q,totalImagesR,Template_count,Template_plot);

save('MPF_Oxford.mat','precision','recall','truePositive',...
    'falsePositive','worstIDCounter','AverageComputeTime','TotalComputeTime','Mem');

end
%-------Next run Nordland--------------------------------------------------
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%--------------------------------------------------------------------------
if Nordland == 1

%clear variables      

%--------------------------------------------------------------------------
%START OF ADJUSTABLE SETTINGS
%Dataset load:
%If set this variable to 1, this makes the ground truth tolerance a +-10
%frame window around the frame id, since the traverses are aligned on
%Nordland.
NordlandGT = 1;           

%Reference Database File Address (provide full path to folder containing images or video):
Ref_folder = 'D:\Windows\Nordland\nordland_summer_images';   
Ref_file_type = '.png';     
Imstart_R = 0;  %start dataset after this many frames or seconds for video.
finalImage_R = 4151;
%finalImage_R = 600;

%Query Database File Address (provide full path to folder containing images or video):
Query_folder = 'D:\Windows\Nordland\nordland_winter_images';
Query_file_type = '.png';
Imstart_Q = 0;  %start dataset after this many frames or seconds for video.
finalImage_Q = 4151;
%finalImage_Q = 600;

Frame_skip = 1;     %Duel use variable: for images, this means use every ith frame
%for videos, this means extract frames out of video at this FPS.

%Ground truth load:
%Load a ground truth correspondance matrix.
GT_file = load('Nordland_GPSMatrix.mat'); %can run the dataset without this
%file, by setting NordlandGT to 1.

%Neural Network load:
datafile = 'D:\MATLAB\MPF_RevisePaper\HybridNet\HybridNet.caffemodel';
protofile = 'D:\MATLAB\MPF_RevisePaper\HybridNet\deploy.prototxt';
% datafile = 'D:\Windows\MATLAB\Caffe_model_vgg16/vgg16_places365.caffemodel';
% protofile = 'D:\Windows\MATLAB\Caffe_model_vgg16/deploy_vgg16_places365.prototxt';
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
plotThresh = 0.4;       %Quality threshold for template plot graph
%(the minimum ROC to trigger a detection in a change of environment novelty)
Rwindow = 10;           %Change this to reflect the approximate distance 
%between frames and the localisation accuracy required.
%Default value (which has been experimentally tested as being a suitable default value) is 20 frames.

%Boolean Flags setting algorithm options
Normalise = 1;          % True: Perform normalisation on CNN feature vectors
qROC_Smooth = 0;        % True: Smooth quality scores over sequence using a moving average
% False: (default) Use raw quality scores for R.O.C. calculation
% Note: As a rule-of-thumb, half Qt for smoothing option.
% Published results use false (default), however the users of this code are
% welcome to use either option to see what works best for your specific
% application.

algSettings = struct('thresh',thresh,'maxSeqLength',maxSeqLength,'minSeqLength',...
    minSeqLength,'obsThresh',obsThresh,'minVelocity',minVelocity,'maxVelocity',...
    maxVelocity,'Qt',Qt,'Rwindow',Rwindow,'qROC_Smooth',qROC_Smooth,...
    'epsilon',epsilon,'plotThresh',plotThresh);

%Image processing method adjustable settings:
Initial_crop = [0 0 0 0];  %crop amount in pixels top, bottom, left, right
Initial_crop = Initial_crop + 1;    %only needed because MATLAB starts arrays from index 1.

SAD_resolution = [64 32];       %width by height
SAD_patchSize = 8;

HOG_resolution = [640 320];     %width by height
HOG_cellSize = [32 32];         

CNN_resolution = [227 227];     %227 for hybridnet and 224 for vgg-16

%actLayer = 24;  %for VGG-16 (note: VGG-16 will run slower than HybridNet)    
actLayer = 15; %default for HybridNet is Conv-5 ReLu layer
%Need to adjust this for the neural network you use.

%END OF ADJUSTABLE SETTINGS
%--------------------------------------------------------------------------

%Run reference traverse
[Template_array1,Template_array2,Template_array3,Template_array4,...
    totalImagesR,Template_count,Template_plot] = ...
    DatabaseLoad(NordlandGT,Ref_folder,Ref_file_type,Imstart_R,...
    Frame_skip,net,actLayer,SAD_resolution,SAD_patchSize,HOG_resolution,...
    HOG_cellSize,CNN_resolution,Initial_crop,Normalise,finalImage_R);

Mem = whos;

%Now run query traverse...
[precision,recall,truePositive,falsePositive,worstIDCounter,AverageComputeTime,TotalComputeTime] = Multi_Process_Fusion_Run(...
    NordlandGT,Ref_folder,Ref_file_type,Query_folder,Query_file_type,...
    Imstart_Q,Imstart_R,Frame_skip,net,actLayer,SAD_resolution,SAD_patchSize,...
    HOG_resolution,HOG_cellSize,CNN_resolution,Initial_crop,Normalise,Template_array1,...
    Template_array2,Template_array3,Template_array4,GT_file,algSettings,...
    finalImage_Q,totalImagesR,Template_count,Template_plot);

save('MPF_Nordland.mat','precision','recall','truePositive',...
    'falsePositive','worstIDCounter','AverageComputeTime','TotalComputeTime','Mem');

end
%-------Finally run Campus----------------------------------------------
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%--------------------------------------------------------------------------
if Campus == 1

%clear variables      

%--------------------------------------------------------------------------
%START OF ADJUSTABLE SETTINGS
%Dataset load:
%Are you using a folder of images or a video file:
NordlandGT = 0;           % 0 = images, 1 = video

%Reference Database File Address (provide full path to folder containing images or video):
Ref_folder = 'D:\Windows\MultiLevel_Campus\multi_level_campus\data\database\images';   
Ref_file_type = '.jpg';     
Imstart_R = 0;  %start dataset after this many frames or seconds for video.
finalImage_R = 9592; %For St Lucia dataset, this is where the first loop ends
%finalImage_R = 960;

%Query Database File Address (provide full path to folder containing images or video):
Query_folder = 'D:\Windows\MultiLevel_Campus\multi_level_campus\data\day\images';   
Query_file_type = '.jpg';
Imstart_Q = 0;  %start dataset after this many frames or seconds for video.
finalImage_Q = 9541;
%finalImage_Q = 960;

Frame_skip = 3;     %Duel use variable: for images, this means use every ith frame
%for videos, this means extract frames out of video at this FPS.

%Ground truth load:
%Load a ground truth correspondance matrix.
GT_file = load('D:\Windows\MultiLevel_Campus\MultiLevelCampus_GPSMatrix.mat');

%Neural Network load:
datafile = 'D:\MATLAB\MPF_RevisePaper\HybridNet\HybridNet.caffemodel';
protofile = 'D:\MATLAB\MPF_RevisePaper\HybridNet\deploy.prototxt';
% datafile = 'D:\Windows\MATLAB\Caffe_model_vgg16/vgg16_places365.caffemodel';
% protofile = 'D:\Windows\MATLAB\Caffe_model_vgg16/deploy_vgg16_places365.prototxt';
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
plotThresh = 0.4;       %Quality threshold for template plot graph
%(the minimum ROC to trigger a detection in a change of environment novelty)
Rwindow = 30;           %Change this to reflect the approximate distance 
%between frames and the localisation accuracy required.
%Default value (which has been experimentally tested as being a suitable default value) is 20 frames.

%Boolean Flags setting algorithm options
Normalise = 1;          % True: Perform normalisation on CNN feature vectors
qROC_Smooth = 0;        % True: Smooth quality scores over sequence using a moving average
% False: (default) Use raw quality scores for R.O.C. calculation
% Note: As a rule-of-thumb, half Qt for smoothing option.
% Published results use false (default), however the users of this code are
% welcome to use either option to see what works best for your specific
% application.

algSettings = struct('thresh',thresh,'maxSeqLength',maxSeqLength,'minSeqLength',...
    minSeqLength,'obsThresh',obsThresh,'minVelocity',minVelocity,'maxVelocity',...
    maxVelocity,'Qt',Qt,'Rwindow',Rwindow,'qROC_Smooth',qROC_Smooth,...
    'epsilon',epsilon,'plotThresh',plotThresh);

%Image processing method adjustable settings:
Initial_crop = [0 0 0 0];  %crop amount in pixels top, bottom, left, right
Initial_crop = Initial_crop + 1;    %only needed because MATLAB starts arrays from index 1.

SAD_resolution = [64 32];       %width by height
SAD_patchSize = 8;

HOG_resolution = [640 320];     %width by height
HOG_cellSize = [32 32];         

CNN_resolution = [227 227];     %224 for vgg-16 and 227 for HybridNet

%actLayer = 24;  %for VGG-16 (note: VGG-16 will run slower than HybridNet)    
actLayer = 15; %default for HybridNet is Conv-5 ReLu layer
%Need to adjust this for the neural network you use.

%END OF ADJUSTABLE SETTINGS
%--------------------------------------------------------------------------

%Run reference traverse
[Template_array1,Template_array2,Template_array3,Template_array4,...
    totalImagesR,Template_count,Template_plot] = ...
    DatabaseLoad(Video_option,Ref_folder,Ref_file_type,Imstart_R,...
    Frame_skip,net,actLayer,SAD_resolution,SAD_patchSize,HOG_resolution,...
    HOG_cellSize,CNN_resolution,Initial_crop,Normalise,finalImage_R);

Mem = whos;

%Now run query traverse...
[precision,recall,truePositive,falsePositive,worstIDCounter,AverageComputeTime,TotalComputeTime] = Multi_Process_Fusion_Run(...
    Video_option,Ref_folder,Ref_file_type,Query_folder,Query_file_type,...
    Imstart_Q,Imstart_R,Frame_skip,net,actLayer,SAD_resolution,SAD_patchSize,...
    HOG_resolution,HOG_cellSize,CNN_resolution,Initial_crop,Normalise,Template_array1,...
    Template_array2,Template_array3,Template_array4,GT_file,algSettings,...
    finalImage_Q,totalImagesR,Template_count,Template_plot);

save('MPF_Campus.mat','precision','recall','truePositive',...
    'falsePositive','worstIDCounter','AverageComputeTime','TotalComputeTime','Mem');

end




