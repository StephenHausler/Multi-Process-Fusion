function [precision,recall,truePositive,falsePositive,worstIDCounter] = Multi_Process_Fusion_Run(varargin)

%global variables:
global Nordland_tunnel_skip
global id2Vid

%Process function inputs
if nargin == 26   
    Video_option = varargin{1};
    Ref_folder = varargin{2};
    Ref_file_type = varargin{3};
    Query_folder = varargin{4};
    Query_file_type = varargin{5};
    Imstart_Q = varargin{6};
    Imstart_R = varargin{7};
    Frame_skip = varargin{8};
    net = varargin{9};
    actLayer = varargin{10};
    SAD_resolution = varargin{11};
    SAD_patchSize = varargin{12};
    HOG_resolution = varargin{13};
    HOG_cellSize = varargin{14};
    Initial_crop = varargin{15};
    Normalise = varargin{16};
    Template_array1 = varargin{17};
    Template_array2 = varargin{18};
    Template_array3 = varargin{19};
    Template_array4 = varargin{20};
    GT_file = varargin{21};
    algSettings = varargin{22};
    finalImage_Q = varargin{23};
    totalImagesR = varargin{24};
    Template_count = varargin{25};
    Template_plot = varargin{26};
else
    error('Incorrect number of inputs to function');
end

%Zeroing Variables
recall_count = zeros(1,length(algSettings.thresh));
error_count = zeros(1,length(algSettings.thresh));
false_negative_count = zeros(1,length(algSettings.thresh));
recall_count2 = 0;
error_count2 = 0;
truePositive = [0 0]; 
falsePositive = [0 0];
Template_count_for_plot = 0;
plot_skip = 0;
worstIDCounter = [0 0 0 0];

O1 = zeros(totalImagesR,1);
O2 = zeros(totalImagesR,1);
O3 = zeros(totalImagesR,1);
O4 = zeros(totalImagesR,1);
T = zeros(totalImagesR);
worstIDArray = zeros(1,totalImagesR);

figure 
hold on
    
%Recall route--------------------------------------------------------------
if Video_option == 1
    vidQ = VideoReader(fullfile(Query_folder,Query_file_type));
    skipIndices = [1290 2040; 2210 2355; 2500 2660; 3400 3670; 5050 5460; 6060 6220]; %if using Nordland
    Imcounter_Q = Imstart_Q;
    totalImagesQ = 0;
    while Imcounter_Q < finalIndex_Q
        if (Nordland_tunnel_skip == 1) && ((Imcounter_Q > skipIndices(1,1)) && (Imcounter_Q < skipIndices(1,2))) 
            Imcounter_Q = Imcounter_Q + Frame_skip;
        elseif (Nordland_tunnel_skip == 1) && ((Imcounter_Q > skipIndices(2,1)) && (Imcounter_Q < skipIndices(2,2))) 
            Imcounter_Q = Imcounter_Q + Frame_skip;
        elseif (Nordland_tunnel_skip == 1) && ((Imcounter_Q > skipIndices(3,1)) && (Imcounter_Q < skipIndices(3,2))) 
            Imcounter_Q = Imcounter_Q + Frame_skip;
        elseif (Nordland_tunnel_skip == 1) && ((Imcounter_Q > skipIndices(4,1)) && (Imcounter_Q < skipIndices(4,2))) 
            Imcounter_Q = Imcounter_Q + Frame_skip;
        elseif (Nordland_tunnel_skip == 1) && ((Imcounter_Q > skipIndices(5,1)) && (Imcounter_Q < skipIndices(5,2)))
            Imcounter_Q = Imcounter_Q + Frame_skip;
        elseif (Nordland_tunnel_skip == 1) && ((Imcounter_Q > skipIndices(6,1)) && (Imcounter_Q < skipIndices(6,2)))
            Imcounter_Q = Imcounter_Q + Frame_skip;
        else
            vidQ.CurrentTime = Imcounter_Q;
            totalImagesQ = totalImagesQ + 1;
            
            Im = readFrame(vidQ);
            
            if totalImagesQ == 1
                %Some pre-allocations:
                diffVector1 = zeros(1,Template_count);
                diffVector2 = zeros(1,Template_count);
                diffVector3 = zeros(1,Template_count);
                diffVector4 = zeros(1,Template_count);
                Template_count_for_plot = Template_count;
                %Transition matrix:
                for j = 1:totalImagesR
                    for k = 1:Template_count
                        if ((k-j) >= algSettings.minVelocity) && ((k-j) <= algSettings.maxVelocity) 
                            T(j,k) = 1; %most likely the robot will stay in place or move forward.
                        else
                            T(j,k) = 0.001; %what is the probability that the robot will take a different route?
                        end
                    end
                end
            end
            
            subplot(2,2,1,'replace');
            image(Im);
            title('Current View');
            
            sz = size(Im);
            Im = Im(Initial_crop(1):(sz(1)-Initial_crop(2)),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
            
            Im1 = imresize(Im,[227 227],'lanczos3');    %for CNN
            Im2 = rgb2gray(Im);
            Im3 = imresize(Im2,[HOG_resolution(2) HOG_resolution(1)],'lanczos3'); %downsize for HOG
            Im4 = imresize(Im2,[SAD_resolution(2) SAD_resolution(1)],'lanczos3'); %downsize for SAD
            
            sum_array1 = CNN_Create_Template(net,Im1,actLayer);         %CNN
            sum_array2 = CNN_Create_Template_Dist(net,Im1,actLayer);    %CNN-Dist
            
            sum_array3 = extractHOGFeatures(Im3,'CellSize',[HOG_cellSize(1) HOG_cellSize(2)]);  %HOG
            
            sum_array4 = zeros(1,size(Im4,1)*size(Im4,2),'int8'); %SAD
            Im4P = patchNormalizeHMM(Im4,SAD_patchSize,0,0);
            sum_array4(1,:) = Im4P(:);
    
            sArray_sz1 = size(sum_array1);
            sArray_sz2 = size(sum_array2);
            
            if Normalise == 1
                sumArrayStore(totalImagesQ,:) = sum_array1;
                
                Q_fAv = mean(sumArrayStore,1);
                
                if totalImagesQ == 1
                    Q_fSt = ones(1,sArray_sz1(2));
                else
                    Q_fSt = std(sumArrayStore,1);
                end
                
                %Now normalise all the features in the current scene:
                for j = 1:sArray_sz1(2)
                    if Q_fSt(j) == 0
                        sum_array1(j) = 0;
                    else
                        sum_array1(j) = (sum_array1(j) - Q_fAv(j))/Q_fSt(j);
                    end
                end
            end
            
            for k = 1:Template_count
                distV = zeros(1,sArray_sz2(2));
                %for sumArray2
                for j = 1:sArray_sz2(2)
                    distV(j) = ( ( Template_array2(1,j,k) - sum_array2(1,j) ).^2 ) +...
                    ( Template_array2(2,j,k) - sum_array2(2,j) ).^2;
                end   
                dist = sqrt(distV);
                diffVector2(k) = sum(dist)/sArray_sz2(2);
            end
            
            D = abs(Template_array4 - sum_array4);
            diffVector4 = sum(D,2);
            diffVector4 = diffVector4';
            
            G_sumArray1 = gpuArray(sum_array1);  %CNN
            G_sumArray3 = gpuArray(sum_array3);  %HOG
            
            G_templateArray1 = gpuArray(Template_array1);
            G_templateArray3 = gpuArray(Template_array3);
            
            G_diffVector1 = gpuArray(zeros(1,Template_count));
            G_diffVector3 = gpuArray(zeros(1,Template_count));
            
            G_diffVector1 = pdist2(G_sumArray1,G_templateArray1,'cosine');
            G_diffVector3 = pdist2(G_sumArray3,G_templateArray3,'cosine');
            
            diffVector1 = gather(G_diffVector1);
            diffVector3 = gather(G_diffVector3);
%--------------------------------------------------------------------------            
            mx1 = max(diffVector1); mx2 = max(diffVector2); mx3 = max(diffVector3); mx4 = max(diffVector4);
            df1 = mx1 - min(diffVector1); df2 = mx2 - min(diffVector2); df3 = mx3 - min(diffVector3); df4 = mx4 - min(diffVector4);
            
            for k = 1:Template_count
                O_diff = ((mx1 - diffVector1(k))/df1)-algSettings.epsilon;
                O_diff2 = ((mx2 - diffVector2(k))/df2)-algSettings.epsilon;
                O_diff3 = ((mx3 - diffVector3(k))/df3)-algSettings.epsilon;
                O_diff4 = ((mx4 - diffVector4(k))/df4)-algSettings.epsilon;
                if O_diff < algSettings.obsThresh
                    O1(totalImagesQ,k) = algSettings.epsilon;
                else
                    O1(totalImagesQ,k) = O_diff;
                end
                if O_diff2 < algSettings.obsThresh
                    O2(totalImagesQ,k) = algSettings.epsilon; 
                else
                    O2(totalImagesQ,k) = O_diff2;
                end            
                if O_diff3 < algSettings.obsThresh
                    O3(totalImagesQ,k) = algSettings.epsilon; 
                else
                    O3(totalImagesQ,k) = O_diff3;
                end  
                if O_diff4 < algSettings.obsThresh
                    O4(totalImagesQ,k) = algSettings.epsilon;
                else
                    O4(totalImagesQ,k) = O_diff4;
                end
            end
            %Find the worst observations for the current image
            [worstID,~] = findWorstID(O1,O2,O3,O4,totalImagesQ);
            worstIDCounter(worstID) = worstIDCounter(worstID) + 1;
            worstIDArray(totalImagesQ) = worstID;   
%--------------------------------------------------------------------------            
            %Use the Viterbi algorithm to find the optimal path through the matrix of recent difference vectors.
            if totalImagesQ > algSettings.maxSeqLength
                S = (totalImagesQ - algSettings.maxSeqLength + 1):totalImagesQ;
                
                [seq,quality,newSeqLength] = viterbi_Smart_Dynamic_Features...
                    (S,T,O1,O2,O3,O4,algSettings.minSeqLength,...
                    algSettings.Rwindow,worstIDArray,algSettings.qROC_Smooth,algSettings.Qt);
            
                quality = quality/newSeqLength;
                
                id = seq(newSeqLength);
%--------------------------------------------------------------------------                
                %loop through every threshold to generate PR curve.
                for thresh_counter = 1:length(algSettings.thresh)
                    if quality > algSettings.thresh(thresh_counter)
                        if (Nordland_tunnel_skip == 1) %on Nordland, the traverses are aligned
                            %no 'new scenes' should be found in Nordland
                            false_negative_count(thresh_counter) = false_negative_count(thresh_counter) + 1;
                        else  %provided GPS mat file
                            if sum(GT_file.GPSMatrix(Imstart_Q+totalImagesQ)) == 0
                                %true negative
                            else
                                false_negative_count(thresh_counter) = false_negative_count(thresh_counter) + 1;
                            end
                        end
                    else
                        if (Nordland_tunnel_skip == 1) %on Nordland, the traverses are aligned
                            if (totalImagesQ > (id-11)) && (totalImagesQ < (id+11))  %up to 10 frames out in either direction
                                recall_count(thresh_counter) = recall_count(thresh_counter) + 1;
                            else
                                error_count(thresh_counter) = error_count(thresh_counter) + 1;
                            end
                        else   %provided GPS mat file
                            if (GT_file.GPSMatrix(Imstart_R+id,Imstart_Q+totalImagesQ)==1)
                                %true positive
                                recall_count(thresh_counter) = recall_count(thresh_counter) + 1;
                            else  %false positive
                                error_count(thresh_counter) = error_count(thresh_counter) + 1;
                            end
                        end
                    end
                end
%--------------------------------------------------------------------------          
                %Now run second set of GT code, to generate template plot
                if quality > algSettings.plotThresh    
                    Template_count_for_plot = Template_count_for_plot + 1;
                    Template_plot(Template_count_for_plot,1) = totalImagesR + totalImagesQ;
                    Template_plot(Template_count_for_plot,2) = Template_count_for_plot;
                else                         
                    if (Nordland_tunnel_skip == 1) %on Nordland, the traverses are aligned
                        if (totalImagesQ > (id-11)) && (totalImagesQ < (id+11))   %up to 10 frames out in either direction
                            recall_count2 = recall_count2 + 1;
                            truePositive(recall_count2,1) = totalImagesR + totalImagesQ;
                            truePositive(recall_count2,2) = id;
                        else
                            error_count2 = error_count2 + 1;
                            falsePositive(error_count2,1) = totalImagesR + totalImagesQ;
                            falsePositive(error_count2,2) = id;
                        end
                    else   %provided GPS mat file
                        if (GT_file.GPSMatrix(Imstart_R+id,Imstart_Q+totalImagesQ)==1)
                            recall_count2 = recall_count2 + 1;
                            truePositive(recall_count2,1) = totalImagesR + totalImagesQ;
                            truePositive(recall_count2,2) = id;
                        else
                            error_count2 = error_count2 + 1;
                            falsePositive(error_count2,1) = totalImagesR + totalImagesQ;
                            falsePositive(error_count2,2) = id;
                        end
                    end 
                end
%--------------------------------------------------------------------------                                  
                Nord_R.CurrentTime = id2Vid(id);
                Im_compare = readFrame(Nord_R);
                subplot(2,2,3,'replace');
                image(Im_compare);
                title('Matched Scene');
                    
                plot_skip = plot_skip + 1;  %prevents compuational slow down
                if plot_skip > 9
                    plot_skip = 0;
                    subplot(2,2,2);
                    title('Template Graph');
                    xlabel('Frame Number');
                    ylabel('Template Number');
                    hold on
                    if ~exist('truePositive','var')
                        
                    else
                        plot(truePositive(:,1),truePositive(:,2),'sg');
                        hold on
                    end
                    if ~exist('falsePositive','var')
                        
                    else
                        plot(falsePositive(:,1),falsePositive(:,2),'sr');
                        hold on
                    end
                    plot(Template_plot(:,1),Template_plot(:,2),'sr');
                end
                drawnow;
            end
            Imcounter_Q = Imcounter_Q + Frame_skip;
        end 
    end 
%--------------------------------------------------------------------------
else        %Not Video    
%--------------------------------------------------------------------------
Query_file_type = strcat('*',Query_file_type);
fQ = dir(fullfile(Query_folder,Query_file_type));

Imcounter_Q = Imstart_Q;
fQ2 = struct2cell(fQ);
filesQ = sort_nat(fQ2(1,:));
i = 1;

while((Imcounter_Q+1) <= finalImage_Q)
    filenamesQ{i} = filesQ(Imcounter_Q+1);
    Imcounter_Q = Imcounter_Q + Frame_skip;
    i=i+1;
end

totalImagesQ = length(filenamesQ);

Ref_file_type = strcat('*',Ref_file_type);
fR = dir(fullfile(Ref_folder,Ref_file_type));

Imcounter_R = Imstart_R;
fR2 = struct2cell(fR);
filesR = sort_nat(fR2(1,:));
i = 1;

while((Imcounter_R+1) <= length(filesR))
    filenamesR{i} = filesR(Imcounter_R+1);
    Imcounter_R = Imcounter_R + Frame_skip;
    i=i+1;
end
    
for ii = 1:totalImagesQ
    if ii == 1  %first time in recall route initialise the transition matrix.
        %Some pre-allocations:
        diffVector1 = zeros(1,Template_count);
        diffVector2 = zeros(1,Template_count);
        diffVector3 = zeros(1,Template_count);
        diffVector4 = zeros(1,Template_count);
        Template_count_for_plot = Template_count;
        %Transition matrix:
        for j = 1:totalImagesR 
            for k = 1:Template_count
                if ((k-j) >= algSettings.minVelocity) && ((k-j) <= algSettings.maxVelocity) 
                    T(j,k) = 1; %most likely the robot will stay in place or move forward.
                else
                    T(j,k) = 0.001; %what is the probability that the robot will take a different route?
                end
            end
        end
    end
    
    Im = imread(char(fullfile(fQ(1).folder,filenamesQ{ii})));
    
    subplot(2,2,1,'replace');
    image(Im);
    title('Current View');
    
    sz = size(Im);
    Im = Im(Initial_crop(1):(sz(1)-Initial_crop(2)),Initial_crop(3):(sz(2)-Initial_crop(4)),:);

    Im1 = imresize(Im,[227 227],'lanczos3');    %for CNN
    Im2 = rgb2gray(Im);
    Im3 = imresize(Im2,[HOG_resolution(2) HOG_resolution(1)],'lanczos3');     %downsize for HOG
    Im4 = imresize(Im2,[SAD_resolution(2) SAD_resolution(1)],'lanczos3');     %downsize for SAD
    
    sum_array1 = CNN_Create_Template(net,Im1,actLayer);         %CNN
    sum_array2 = CNN_Create_Template_Dist(net,Im1,actLayer);    %CNN-Dist
    
    sum_array3 = extractHOGFeatures(Im3,'CellSize',[HOG_cellSize(1) HOG_cellSize(2)]);  %HOG
    
    sum_array4 = zeros(1,size(Im4,1)*size(Im4,2),'int8'); %SAD
    Im4P = patchNormalizeHMM(Im4,SAD_patchSize,0,0);
    sum_array4(1,:) = Im4P(:);
    
    sArray_sz1 = size(sum_array1);
    sArray_sz2 = size(sum_array2);
    
    if Normalise == 1
        sumArrayStore(ii,:) = sum_array1;
        
        Q_fAv = mean(sumArrayStore,1);
        
        if ii == 1
            Q_fSt = ones(1,sArray_sz1(2));
        else
            Q_fSt = std(sumArrayStore,1);
        end
        
        %Now normalise all the features in the current scene:
        for j = 1:sArray_sz1(2)
            if Q_fSt(j) == 0
                sum_array1(j) = 0;
            else
                sum_array1(j) = (sum_array1(j) - Q_fAv(j))/Q_fSt(j);
            end
        end
    end
            
    for k = 1:Template_count  %For CNN-Dist
        distV = zeros(1,sArray_sz2(2));
        %for sumArray2
        for j = 1:sArray_sz2(2)
            distV(j) = ( ( Template_array2(1,j,k) - sum_array2(1,j) ).^2 ) +...
            ( Template_array2(2,j,k) - sum_array2(2,j) ).^2;
        end   
        dist = sqrt(distV);
        diffVector2(k) = sum(dist)/sArray_sz2(2);
    end
    
    D = abs(Template_array4 - sum_array4);
    diffVector4 = sum(D,2);
    diffVector4 = diffVector4';
    
    G_sumArray1 = gpuArray(sum_array1); %CNN
    G_sumArray3 = gpuArray(sum_array3); %HOG
    
    G_templateArray1 = gpuArray(Template_array1);
    G_templateArray3 = gpuArray(Template_array3);
    
    G_diffVector1 = gpuArray(zeros(1,Template_count));
    G_diffVector3 = gpuArray(zeros(1,Template_count));
    
    G_diffVector1 = pdist2(G_sumArray1,G_templateArray1,'cosine');
    G_diffVector3 = pdist2(G_sumArray3,G_templateArray3,'cosine');
    
    diffVector1 = gather(G_diffVector1);
    diffVector3 = gather(G_diffVector3);
%--------------------------------------------------------------------------
%Compute the observation matricies using the cosine distance scores
%This normalisation calc is prone to issues if significant outliers are
%present in the reference traverse.
    mx1 = max(diffVector1); mx2 = max(diffVector2); mx3 = max(diffVector3); mx4 = max(diffVector4);
    df1 = mx1 - min(diffVector1); df2 = mx2 - min(diffVector2); df3 = mx3 - min(diffVector3); df4 = mx4 - min(diffVector4);
    
    for k = 1:Template_count
        O_diff = ((mx1 - diffVector1(k))/df1)-algSettings.epsilon;  %normalise to range 0.001 to 0.999    
        O_diff2 = ((mx2 - diffVector2(k))/df2)-algSettings.epsilon;
        O_diff3 = ((mx3 - diffVector3(k))/df3)-algSettings.epsilon;
        O_diff4 = ((mx4 - diffVector4(k))/df4)-algSettings.epsilon;
        if O_diff < algSettings.obsThresh 
            O1(ii,k) = algSettings.epsilon; 
        else
            O1(ii,k) = O_diff;
        end 
        if O_diff2 < algSettings.obsThresh
            O2(ii,k) = algSettings.epsilon; 
        else
            O2(ii,k) = O_diff2;
        end            
        if O_diff3 < algSettings.obsThresh
            O3(ii,k) = algSettings.epsilon; 
        else
            O3(ii,k) = O_diff3;
        end         
        if O_diff4 < algSettings.obsThresh
            O4(ii,k) = algSettings.epsilon;
        else
            O4(ii,k) = O_diff4;
        end
    end
    %Find the worst observations for the current image
    [worstID] = findWorstID(O1,O2,O3,O4,ii);
    worstIDCounter(worstID) = worstIDCounter(worstID) + 1;
    worstIDArray(ii) = worstID;
    
%--------------------------------------------------------------------------
%Use the Viterbi algorithm to find the optimal path through the matrix of recent difference vectors.                
    if ii > algSettings.maxSeqLength

        S = (ii-algSettings.maxSeqLength+1):ii;  %length(S) should be equal to maxSeqLength.

        [seq,quality,newSeqLength] = viterbi_Smart_Dynamic_Features(S,T,O1,O2,O3,O4,...
            algSettings.minSeqLength,algSettings.Rwindow,worstIDArray,...
            algSettings.qROC_Smooth,algSettings.Qt);        
        
        quality = quality/newSeqLength;
            
        id = seq(newSeqLength);     
%--------------------------------------------------------------------------
        %loop through every threshold to generate the PR curve.
        for thresh_counter = 1:length(algSettings.thresh)
            if quality > algSettings.thresh(thresh_counter)
                if sum(GT_file.GPSMatrix(Imstart_Q+ii,:))==0
                    %true negative
                else
                    %false negative
                    false_negative_count(thresh_counter) = false_negative_count(thresh_counter) + 1;
                end
            else
                if (GT_file.GPSMatrix(Imstart_R+id,Imstart_Q+ii)==1)
                    %true positive
                    recall_count(thresh_counter) = recall_count(thresh_counter) + 1;
                else  %false positive
                    error_count(thresh_counter) = error_count(thresh_counter) + 1;
                end
            end
        end        
%--------------------------------------------------------------------------        
            %Now run second set of GT code, to generate template plot
            if quality > algSettings.plotThresh    
                Template_count_for_plot = Template_count_for_plot + 1;
                Template_plot(Template_count_for_plot,1) = totalImagesR + ii;
                Template_plot(Template_count_for_plot,2) = Template_count_for_plot;     
            else    
                if (GT_file.GPSMatrix(Imstart_R+id,Imstart_Q+ii)==1)
                    recall_count2 = recall_count2 + 1;
                    truePositive(recall_count2,1) = ii + totalImagesR;
                    truePositive(recall_count2,2) = id;
                else
                    error_count2 = error_count2 + 1;
                    falsePositive(error_count2,1) = ii + totalImagesR;
                    falsePositive(error_count2,2) = id;
                end
            end 
%--------------------------------------------------------------------------                      
            subplot(2,2,3,'replace');
            Im_compare = imread(char(fullfile(fR(1).folder,filenamesR{id})));
            image(Im_compare);
            title('Matched Scene');
                       
            plot_skip = plot_skip+1;
            if plot_skip > 9
                plot_skip = 0;
                subplot(2,2,2);
                title('Template Graph');
                xlabel('Frame Number');
                ylabel('Template Number');
                hold on 
                if ~exist('truePositive','var')

                else
                    plot(truePositive(:,1),truePositive(:,2),'sg');
                    hold on
                end
                if ~exist('falsePositive','var')

                else
                    plot(falsePositive(:,1),falsePositive(:,2),'sr');
                    hold on    
                end
                plot(Template_plot(:,1),Template_plot(:,2),'sr');
            end   
            drawnow;             
%--------------------------------------------------------------------------       
    end
end
end
%--------------------------------------------------------------------------
for thresh_counter = 1:length(algSettings.thresh)
    %Recall = true positives / (true positives + false negatives)
    recall(thresh_counter) = recall_count(thresh_counter)/(recall_count(thresh_counter) + false_negative_count(thresh_counter));
    %Precision = true positives / (true positives + false positives)
    precision(thresh_counter) = recall_count(thresh_counter)/(recall_count(thresh_counter) + error_count(thresh_counter));
end
%plot the precision-recall curve:
figure
plot(recall,precision,'-r');
title('Precision Recall Curve');
xlabel('Recall');
ylabel('Precision');
xlim([0 1]);
ylim([0 1]);
grid on
      
end    
        