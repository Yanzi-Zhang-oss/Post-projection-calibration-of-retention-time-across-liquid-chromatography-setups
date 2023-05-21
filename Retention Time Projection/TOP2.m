% clear
% %加载实验数据和外部数据
% [RT] = excel_import();
% 
% [outside_RT] = excel_import_other_data();
% 
% % GROUP_NUM = 4;
% % RT = test44_SOM82;
% % 
% % 
% % [chem_symbol] = SOM_train(RT,GROUP_NUM);
% % load SOM_ANN.mat
% % 
% % [train,validation,test] = splitData(RT,chem_symbol,...
% %                                 0.2,0.4,0.4);
% % 
% %                             

train = CNN_Cal_train9;
test = CNN_Train_train9;
                            
                            
                            
                            
                            
% [chem_total_num,RT_num] = size(RT);

% in = [RT(1:50,1);RT(1:50,2);RT(51,1)];
% response = RT(51,2);

%生成测试数据
%输入组成  Art1（50个）,Art2（50个）,Brt1(单个)
%输出组成  Brt2（单个）
data_mat = [];%变量定义，初始化为空数据
%symbol_num = 50;



try
    parpool(4)
catch
    disp('pool already start')
end

%[RT] = rt_reshape(RT);
% 
% train = xunlian55_SOM82;
% test = test55_SOM82;
[symbol_num,RT_num] = size(train);
[chem_total_num,~] = size(test);

for i = 1:1: chem_total_num
    parfor ( j = 1:RT_num)
        for k = 1:1:RT_num
            if j ~= k
                temp = [train(1:symbol_num,j);train(1:symbol_num,k);...
                    test(i,j);test(i,k);];
                data_mat = [data_mat,temp];
                disp(i)
            end
        end
    end
end

in = data_mat(1:symbol_num*2+1,:);
response = data_mat(symbol_num*2+2,:);

[score] = ANN_train_and_test(in,response);
