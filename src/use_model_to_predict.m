function use_model_to_predict (train_model,test1,test2)

try
    if class(train_model) == 'network'
        y_predict = train_model(test1');
        y_predict = y_predict';
    end
end

try
    if  class(train_model) == 'struct'
        y_predict = train_model.predictFcn(test1);
    end
end



sub = y_predict - test2;

plot(y_predict,'DisplayName','yfit');hold on;plot(test2,'DisplayName','CNN_Test2');hold off;
legend('预测值','测试值')

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
rmse = sqrt(mean((y_predict-test2).^2));
%R2
R2_mat = corrcoef(y_predict,test2);

R2 = R2_mat(1,2);

%clearvars  -except yfit sub range_max range_min rmse R2 PI_99_low PI_99_high PI_95_low PI_95_high

save A_TEST_result.mat y_predict  sub  range_max  range_min  rmse  R2  PI_99_low  PI_99_high  PI_95_low  PI_95_high   test2


end