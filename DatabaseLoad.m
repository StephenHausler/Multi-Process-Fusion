%--------------------------------------------------------------------------
%   Title: Multi-Process Fusion
%   Author: Stephen Hausler
%   
%   Open Source Code, requires MATLAB with Neural Network Toolbox
%   Refer to LICENSES.txt for license to this source code and all 3rd party
%   licences.
%--------------------------------------------------------------------------

function [Template_array1,Template_array2,Template_array3,Template_array4,...
    totalImagesR,Template_count,Template_plot] = DatabaseLoad(varargin)

Template_count = 0;
global Nordland_tunnel_skip
global id2Vid

if nargin == 14
    Video_option = varargin{1};
    Ref_folder = varargin{2};
    Ref_file_type = varargin{3};
    Imstart_R = varargin{4};
    Frame_skip = varargin{5};
    net = varargin{6};
    actLayer = varargin{7};
    SAD_resolution = varargin{8};
    SAD_patchSize = varargin{9};
    HOG_resolution = varargin{10};
    HOG_cellSize = varargin{11};
    Initial_crop = varargin{12};
    Normalise = varargin{13};
    finalImage_R = varargin{14};
else
    error('invalid number of inputs');
end

if Video_option == 0
    Ref_file_type = strcat('*',Ref_file_type);
    fR = dir(fullfile(Ref_folder,Ref_file_type));

    Imcounter_R = Imstart_R;
    fR2 = struct2cell(fR);
    filesR = sort_nat(fR2(1,:));
    i = 1;
    
    while((Imcounter_R+1) <= finalImage_R)
        filenamesR{i} = filesR(Imcounter_R+1);
        Imcounter_R = Imcounter_R + Frame_skip;
        i=i+1;
    end
    
    totalImagesR = length(filenamesR);
    
    for i = 1:totalImagesR
        Im = imread(char(fullfile(fR(1).folder,filenamesR{i})));
        sz = size(Im);
        Im = Im((Initial_crop(1):(sz(1)-Initial_crop(2))),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
        
        Im1 = imresize(Im,[227 227],'lanczos3');  %for CNN
        Im2 = rgb2gray(Im);
        Im3 = imresize(Im2,[HOG_resolution(2) HOG_resolution(1)],'lanczos3'); %downsize for HOG
        Im4 = imresize(Im2,[SAD_resolution(2) SAD_resolution(1)],'lanczos3'); %downsize for SAD
        
        template1 = CNN_Create_Template(net,Im1,actLayer);      %CNN
        template2 = CNN_Create_Template_Dist(net,Im1,actLayer); %CNN-Dist
        template3 = extractHOGFeatures(Im3,'CellSize',[HOG_cellSize(1) HOG_cellSize(2)]); %HOG
        template4 = zeros(1,size(Im4,1)*size(Im4,2),'int8');    %SAD
        Im4P = patchNormalizeHMM(Im4,SAD_patchSize,0,0);
        template4(1,:) = Im4P(:);
        
        %Now store in template matrix:
        Template_count = Template_count + 1;
        Template_plot(Template_count,1) = i;
        Template_plot(Template_count,2) = Template_count;
        
        Template_array1(Template_count,:) = template1;
        Template_array2(1,:,Template_count) = template2(1,:);
        Template_array2(2,:,Template_count) = template2(2,:);  
        Template_array3(Template_count,:) = template3;
        Template_array4(Template_count,:) = template4;
    end
else
    vidR = VideoReader(strcat(Ref_folder,Ref_file_type));
    skipIndices = [1290 2040; 2210 2355; 2500 2660; 3400 3670; 5050 5460; 6060 6220]; %if using Nordland
    Imcounter_R = Imstart_R;
    totalImagesR = 0;
    while Imcounter_R < finalImage_R
        if (Nordland_tunnel_skip == 1) && ((Imcounter_R > skipIndices(1,1)) && (Imcounter_R < skipIndices(1,2)))
            Imcounter_R = Imcounter_R + Frame_skip;
        elseif (Nordland_tunnel_skip == 1) && ((Imcounter_R > skipIndices(2,1)) && (Imcounter_R < skipIndices(2,2)))
            Imcounter_R = Imcounter_R + Frame_skip;
        elseif (Nordland_tunnel_skip == 1) && ((Imcounter_R > skipIndices(3,1)) && (Imcounter_R < skipIndices(3,2)))
            Imcounter_R = Imcounter_R + Frame_skip;
        elseif (Nordland_tunnel_skip == 1) && ((Imcounter_R > skipIndices(4,1)) && (Imcounter_R < skipIndices(4,2)))
            Imcounter_R = Imcounter_R + Frame_skip;
        elseif (Nordland_tunnel_skip == 1) && ((Imcounter_R > skipIndices(5,1)) && (Imcounter_R < skipIndices(5,2)))
            Imcounter_R = Imcounter_R + Frame_skip;
        elseif (Nordland_tunnel_skip == 1) && ((Imcounter_R > skipIndices(6,1)) && (Imcounter_R < skipIndices(6,2)))
            Imcounter_R = Imcounter_R + Frame_skip;
        else
            vidR.CurrentTime = Imcounter_R;
            totalImagesR = totalImagesR + 1;    
            Im = readFrame(vidR);         %this should be a 1080 heigh by 1920 wide image
            
            %create a special array to store the exact correspondences between
            %imcounter_R and totalImagesR.
            id2Vid(totalImagesR) = Imcounter_R;
            
            sz = size(Im);
            Im = Im(Initial_crop(1):(sz(1)-Initial_crop(2)),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
            
            Im1 = imresize(Im,[227 227],'lanczos3');  %for CNN
            Im2 = rgb2gray(Im);
            Im3 = imresize(Im2,[HOG_resolution(2) HOG_resolution(1)],'lanczos3'); %downsize for HOG
            Im4 = imresize(Im2,[SAD_resolution(2) SAD_resolution(1)],'lanczos3'); %downsize for SAD
            
            template1 = CNN_Create_Template(net,Im1,actLayer);      %CNN
            template2 = CNN_Create_Template_Dist(net,Im1,actLayer); %CNN-Dist
            template3 = extractHOGFeatures(Im3,'CellSize',[HOG_cellSize(1) HOG_cellSize(2)]); %HOG
            template4 = zeros(1,size(Im4,1)*size(Im4,2),'int8');    %SAD
            Im4P = patchNormalizeHMM(Im4,SAD_patchSize,0,0);
            template4(1,:) = Im4P(:);
            
            %Now store in template matrix:
            Template_count = Template_count + 1;
            Template_plot(Template_count,1) = totalImagesR;
            Template_plot(Template_count,2) = Template_count;
            
            Template_array1(Template_count,:) = template1;
            Template_array2(1,:,Template_count) = template2(1,:);
            Template_array2(2,:,Template_count) = template2(2,:);
            Template_array3(Template_count,:) = template3;
            Template_array4(Template_count,:) = template4;
            
            Imcounter_R = Imcounter_R + Frame_skip;
        end
    end
end
if Normalise == 1
    fAv = mean(Template_array1,1);
    fSt = std(Template_array1,1);
    sz = size(Template_array1);
    
    %Now perform normalisation on every element in the template arrays.
    for j = 1:sz(2)
        if fSt(j) == 0 
            Template_array1(:,j) = 0;
        else
            Template_array1(:,j) = (Template_array1(:,j) - fAv(j))/fSt(j);
        end
    end   
end

end

