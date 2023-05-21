function [] = ...
            trainModel_top_ANN(CNN_Cal1, CNN_Cal2)
rng('default')  % For reproducibility
[yfit,Model_ANN] = fitnet_train(CNN_Cal1,CNN_Cal2);

sub = yfit - CNN_Cal2;

plot(yfit,'DisplayName','yfit');hold on;plot(CNN_Cal2,'DisplayName','CNN_Test2');hold off;
legend('预测值','实际值')

%PI  %是预测值减去真实值的差值的置信区间
[n,~] = size(sub);
sub_average = mean(sub);
sub_std = std(sub);


PI_95_low = sub_average - 1.96*sub_std/sqrt(n);
PI_95_high = sub_average + 1.96*sub_std/sqrt(n); 
PI_99_low = sub_average - 2.58*sub_std/sqrt(n);
PI_99_high = sub_average + 2.58*sub_std/sqrt(n); 

%极差
range_max = max(sub);
range_min=min(sub);
%RMSE
rmse = sqrt(mean((yfit-CNN_Cal2).^2));
%R2
R2_mat = corrcoef(yfit,CNN_Cal2);

R2 = R2_mat(1,2);

save A_ANN_result.mat yfit  sub  range_max  range_min  rmse  R2  PI_99_low  PI_99_high  PI_95_low  PI_95_high Model_ANN  CNN_Cal2


end

function [Y,trainedModel_ANN] = fitnet_train(inputs,outputs)
    
	
net = feedforwardnet(10);   %隐藏层为10层
	
% Train the Network
net = train(net,inputs',outputs','useParallel','no');
view(net)
% Test the Network
Y = net(inputs');

trainedModel_ANN = net;

Y = Y';

end