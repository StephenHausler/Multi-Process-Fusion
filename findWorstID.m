function [worstID] = findWorstID(O1,O2,O3,O4,ii,Rwindow)

quality = [0 0 0 0];

oo1 = O1(ii,:)';
oo2 = O2(ii,:)';
oo3 = O3(ii,:)';
oo4 = O4(ii,:)';

%experiment: what if take logs here as well?
log_option = 0;
if log_option == 1
    oo1 = log(oo1);
    oo2 = log(oo2);
    oo3 = log(oo3);
    oo4 = log(oo4);
end
    [min_value_oo1,min_index_oo1] = max(oo1,[],1);  
    [min_value_oo2,min_index_oo2] = max(oo2,[],1);   
    [min_value_oo3,min_index_oo3] = max(oo3,[],1);   
    [min_value_oo4,min_index_oo4] = max(oo4,[],1);   

    window = max(1, min_index_oo1 - Rwindow):min(length(oo1), min_index_oo1 + Rwindow);
    not_window = setxor(1:length(oo1), window);
    min_value_2nd = max(oo1(not_window));
    quality(1) = min_value_oo1 / min_value_2nd;

    window = max(1, min_index_oo2 - Rwindow):min(length(oo2), min_index_oo2 + Rwindow);
    not_window = setxor(1:length(oo2), window);
    min_value_2nd = max(oo1(not_window));
    quality(2) = min_value_oo2 / min_value_2nd;

    window = max(1, min_index_oo3 - Rwindow):min(length(oo3), min_index_oo3 + Rwindow);
    not_window = setxor(1:length(oo3), window);
    min_value_2nd = max(oo3(not_window));
    quality(3) = min_value_oo3 / min_value_2nd;

    window = max(1, min_index_oo4 - Rwindow):min(length(oo4), min_index_oo4 + Rwindow);
    not_window = setxor(1:length(oo4), window);
    min_value_2nd = max(oo4(not_window));
    quality(4) = min_value_oo4 / min_value_2nd;

if log_option == 1
    [~,worstID] = max(quality);
else    
    [~,worstID] = min(quality);
end
