function [S,quality_total,seqLength] = viterbi_Smart_Dynamic_Features(Y,T,...
    O1,O2,O3,O4,minSeqLength,Rwindow,worstIDArray,qROC_Smooth,Qt)

%MultiProcessFusion, compute the worst processing method by sequence and at the
%very beginning of this function.
    
tau = length(Y); [~,kk] = size(O1); 

oo1 = O1(Y,:)';
oo2 = O2(Y,:)';
oo3 = O3(Y,:)';
oo4 = O4(Y,:)';

T = log(T);
oo1L = log(oo1);
oo2L = log(oo2);
oo3L = log(oo3);
oo4L = log(oo4);
%log of a number less than 1 is a negative number  
%log(0.001) = -6.9078 (worst)
%log(0.999) = -0.0010 (best)

%for each element in worstIDArray, modify the choice of observations used.
for i=1:tau
    switch worstIDArray(Y(i))
        case 0
            fullObs(:,i) = oo1L(:,i) + oo2L(:,i) + oo3L(:,i) + oo4L(:,i);
        case 1
            fullObs(:,i) = oo2L(:,i) + oo3L(:,i) + oo4L(:,i);
        case 2
            fullObs(:,i) = oo1L(:,i) + oo3L(:,i) + oo4L(:,i);
        case 3
            fullObs(:,i) = oo1L(:,i) + oo2L(:,i) + oo4L(:,i);
        case 4
            fullObs(:,i) = oo1L(:,i) + oo2L(:,i) + oo3L(:,i);
    end
end
%using logarithms speeds up computations (normally this would be a
%multiplication) and amplifies the quality discrimination by creating a
%non-linearity in the quality scoring.
[min_values,min_indicies] = max(fullObs,[],1);   %max value of each column
         
for i = 1:tau
    window = max(1, min_indicies(i)-Rwindow):min(length(fullObs), min_indicies(i)+Rwindow);
    not_window = setxor(1:length(fullObs), window);
    min_value_2nd = max(fullObs(not_window,i));
    quality(i) = min_values(i) / min_value_2nd;
end
%Option of calculating a moving average quality over the sequence:
%(choice of 'smoothed' or 'not-smoothed' quality ROC).
if qROC_Smooth == 1
    for i = 1:tau
        if i == 1
            quality_av(i) = (quality(i) + quality(i+1))/2;
        elseif i == tau
            quality_av(i) = (quality(i-1) + quality(i))/2;
        else
            quality_av(i) = (quality(i-1) + quality(i) + quality(i+1))/3;
        end
    end
    %Remember, because we took the logarithm of these values, 1 --> 0 and 0 -->
    %infinity. Therefore a smaller ratio is a better quality match.
    for i = 2:(tau-minSeqLength+1)  %20-5+1=16, then qROC has i-1 so max index is 15
        qROC(i-1) = (quality_av(i) - quality_av(i-1)); %this works because we use the logarithm of the quality scores
    end
else  %default method - choice of which method to use depends on multiple factors,
%such as, spacing between images and amount of perceptual aliasing.
    for i = 2:(tau-minSeqLength+1)  
        qROC(i-1) = (quality(i) - quality(i-1)); 
    end
end

[qCompare,seqStart] = min(qROC);   %greatest positive or negative Rate Of Change
%negative ROC is the point the sequence goes from 'bad' to 'good'.

%if no ROC is great enough, set seqStart to 0 (longest sequence length)
if abs(qCompare) < Qt
    seqStart = 0;
end

tau = tau - seqStart;

%To speed the compute time, replace these floating point numbers with
%fixed length integers - multiple by 1000 (for 3 decimal places of precision)
%then convert to int16.
T = int16(T*1000);
oo1L = int16(oo1L*1000);
oo2L = int16(oo2L*1000);
oo3L = int16(oo3L*1000);
oo4L = int16(oo4L*1000);
%this has been found to result in a significant computational speed-up

%initialize Viterbi variables
delta=zeros(kk,tau);  H=zeros(kk,tau); S=zeros(1,tau);
delta = int16(delta);
H(:,1) = 0;

switch worstIDArray(Y(1))
    case 0
        delta(:,1) = oo1L(:,seqStart+1) + oo2L(:,seqStart+1) + oo3L(:,seqStart+1) + oo4L(:,seqStart+1);
    case 1
        delta(:,1) = oo2L(:,seqStart+1) + oo3L(:,seqStart+1) + oo4L(:,seqStart+1);
    case 2
        delta(:,1) = oo1L(:,seqStart+1) + oo3L(:,seqStart+1) + oo4L(:,seqStart+1);
    case 3
        delta(:,1) = oo1L(:,seqStart+1) + oo2L(:,seqStart+1) + oo4L(:,seqStart+1);
    case 4
        delta(:,1) = oo1L(:,seqStart+1) + oo2L(:,seqStart+1) + oo3L(:,seqStart+1);
end       
                
for i = 2:tau
    [delta(:,i),H(:,i)] = max((repmat(delta(:,i-1),1,kk)+T),[],1);
    switch worstIDArray(Y(i))
        case 0
            delta(:,i) = delta(:,i)+oo1L(:,i+seqStart)+oo2L(:,i+seqStart)+oo3L(:,i+seqStart)+oo4L(:,i+seqStart);
        case 1
            delta(:,i) = delta(:,i)+oo2L(:,i+seqStart)+oo3L(:,i+seqStart)+oo4L(:,i+seqStart);
        case 2
            delta(:,i) = delta(:,i)+oo1L(:,i+seqStart)+oo3L(:,i+seqStart)+oo4L(:,i+seqStart);
        case 3
            delta(:,i) = delta(:,i)+oo1L(:,i+seqStart)+oo2L(:,i+seqStart)+oo4L(:,i+seqStart);
        case 4
            delta(:,i) = delta(:,i)+oo1L(:,i+seqStart)+oo2L(:,i+seqStart)+oo3L(:,i+seqStart);
    end
end

[~,S(tau)] = max(delta(:,tau)); 
quality_total = 0;
  
for k=(tau-1):-1:1
  S(k)=H(S(k+1),k+1);
  %quality score generated by the ratio between the delta value at the
  %backtracking route and the next highest number outside a window R around
  %the backtracked route.
  min_idx = S(k+1);
  min_value = delta(S(k+1),k+1);
  min_value = double(min_value);
  window = max(1, min_idx-Rwindow):min(length(delta), min_idx+Rwindow);
  not_window = setxor(1:length(delta), window);
  min_value_2nd = max(delta(not_window,k+1));
  min_value_2nd = double(min_value_2nd);
  quality = min_value / min_value_2nd;
  quality_total = quality_total + quality; 
end

seqLength = tau;
