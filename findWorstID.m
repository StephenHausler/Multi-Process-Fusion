function [worstId] = findWorstID(O1,O2,O3,O4,ii,Rwindow)

oo1 = O1(ii,:);
oo2 = O2(ii,:);
oo3 = O3(ii,:);
oo4 = O4(ii,:);

[~,moo1] = max(oo1); 
[~,moo2] = max(oo2); 
[~,moo3] = max(oo3); 
[~,moo4] = max(oo4); 
%spatial coherence distance values
d12 = abs(moo1 - moo2);
d13 = abs(moo1 - moo3);
d14 = abs(moo1 - moo4);
d23 = abs(moo2 - moo3);
d24 = abs(moo2 - moo4);
d34 = abs(moo3 - moo4);

So1 = 0; So2 = 0; So3 = 0; So4 = 0;
%spatial coherence score per sensor
if d12 <= Rwindow
    So1 = So1 + 1;
    So2 = So2 + 1;
end
if d13 <= Rwindow
    So1 = So1 + 1;
    So3 = So3 + 1;
end 
if d14 <= Rwindow
    So1 = So1 + 1;
    So4 = So4 + 1;
end    
if d23 <= Rwindow
    So2 = So2 + 1;
    So3 = So3 + 1;
end
if d24 <= Rwindow
    So2 = So2 + 1;
    So4 = So4 + 1;
end
if d34 <= Rwindow
    So3 = So3 + 1;
    So4 = So4 + 1;
end    
S = [So1 So2 So3 So4];
%if all are zero or max, then use all...
if (So1 == 0) && (So2 == 0) && (So3 == 0) && (So4 == 0)
    worstId = 0;
elseif (So1 == 3) && (So2 == 3) && (So3 == 3) && (So4 == 3)
    worstId = 0;
else   %note: in the event of a 'tie', i.e. two sensors report a location 
    %and another two sensors report a different location, precedence is 
    %established by the default sensor order So1, So2, So3, So4, which is
    %CNN, CNN-D, HOG, SAD. Default order decided by previous experiments
    %which demonstrates the overall (across multiple datasets) single image
    %processing method performance.
    [~,bestId] = max(S);  %which is the most spatially consistent
    if bestId == 1
        [~,wId] = min([oo2(moo1) oo3(moo1) oo4(moo1)]);
        if wId == 1
            worstId = 2;
        elseif wId == 2
            worstId = 3;
        else
            worstId = 4;
        end
    elseif bestId == 2
        [~,wId] = min([oo1(moo2) oo3(moo2) oo4(moo2)]);
        if wId == 1
            worstId = 1;
        elseif wId == 2
            worstId = 3;
        else
            worstId = 4;
        end
    elseif bestId == 3
        [~,wId] = min([oo1(moo3) oo2(moo3) oo4(moo3)]);
        if wId == 1
            worstId = 1;
        elseif wId == 2
            worstId = 2;
        else
            worstId = 4;
        end
    else
        [~,wId] = min([oo1(moo4) oo2(moo4) oo3(moo4)]);
        if wId == 1
            worstId = 1;
        elseif wId == 2
            worstId = 2;
        else
            worstId = 3;
        end
    end   
end    
end
