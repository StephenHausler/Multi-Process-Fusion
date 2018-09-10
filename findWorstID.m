function [worstID,quality] = findWorstID(O1,O2,O3,O4,ii)

quality = [0 0 0 0];

oo1 = O1(ii,:)';
oo2 = O2(ii,:)';
oo3 = O3(ii,:)';
oo4 = O4(ii,:)';

    [~,min_index_oo1] = max(oo1,[],1);  
    [~,min_index_oo2] = max(oo2,[],1);   
    [~,min_index_oo3] = max(oo3,[],1);   
    [~,min_index_oo4] = max(oo4,[],1);   

	av_index = mean([min_index_oo1 min_index_oo2 min_index_oo3 min_index_oo4]);
	
	quality(1) = abs(min_index_oo1 - av_index);
	quality(2) = abs(min_index_oo2 - av_index);
	quality(3) = abs(min_index_oo3 - av_index);
	quality(4) = abs(min_index_oo4 - av_index);
	
	[~,worstID] = max(quality);
	
end
