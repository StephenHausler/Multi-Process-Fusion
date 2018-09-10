function [template] = CNN_Create_Template_Dist(net,Im,actLayer)

    act = activations(net, Im, actLayer,'OutputAs','channels','ExecutionEnvironment','gpu');

    sz1 = size(act); 

    act1 = reshape(act,[sz1(1) sz1(2) 1 sz1(3)]);

    template=0;

    for j = 1:sz1(3)
        [temp,actx] = max(act1(:,:,1,j));
        [~,acty] = max(temp);
        template(1,j) = actx(acty);
        template(2,j) = acty;
    end

end




