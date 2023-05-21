
function [out_B] = prediction_net(A_1,B_1,A_2)
    load feedforwardRT13_cal52.mat
    [a,b] = size(B_1);
    
    if a == 1
        B_1 = B_1;
    elseif b == 1
        B_1 = B_1';
    end
    [m,n] = size(B_1);
    out_B = zeros(1,n);
    
    
    for i = 1:1:n
        xfit = [];
        xfit = [A_1(:,1);A_2(:,1)];
        out_B(1,i) = net([xfit;B_1(1,i)]);
    end
    
    out_B = out_B';

end