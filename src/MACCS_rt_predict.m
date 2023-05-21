clear
load data_1009.mat

% %补0
% [m,n] = size(MACCS);
% MACCS = [MACCS zeros(m,3)];

MACCS_rt(1:3000) = [];
MACCS(1:3000,:) = [];



[~,~,I] = unique(MACCS_rt,'stable');


[m,n] = size(MACCS);

chem = strings(m,1);


[m,~] = size(MACCS);

[train,test] = crossvalind('LeaveMOut',m,m*0.3);

%[~,I] = sort(rt);

[~,~,I] = unique(MACCS_rt,'stable');

%I = I - (max(I) + min(I))/2;

% textDataTrain = chem(train);
% textDataValidation = chem(test);

XTrain = MACCS(train,:);
XValidation = MACCS(test,:);
[m_XT,~] = size(XTrain);
[m_XV,~] = size(XValidation);

XTrain = reshape(XTrain,m_XT,1,166,1);
XValidation = reshape(XValidation,m_XV,1,166,1);

XTrain = permute(XTrain,[3 4 2 1]);
XValidation = permute(XValidation,[3 4 2 1]);

YTrain = MACCS_rt(train);
YValidation = MACCS_rt(test);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lgraph = layerGraph();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tempLayers = [
    imageInputLayer([166 1 1],"Name","imageinput")
    convolution2dLayer([3 1],16,"Name","conv_1")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],16,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    convolution2dLayer([3 1],16,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding",[0 1 0 1],"Stride",[3 3])
    convolution2dLayer([3 1],32,"Name","conv_4")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],32,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    convolution2dLayer([3 1],32,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2_1","Padding",[0 1 0 1],"Stride",[3 3])
    convolution2dLayer([3 1],64,"Name","conv_7_1")
    batchNormalizationLayer("Name","batchnorm_7_1")
    reluLayer("Name","relu_7_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],64,"Name","conv_8_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8_1")
    reluLayer("Name","relu_8_1")
    convolution2dLayer([3 1],64,"Name","conv_9_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9_1")
    reluLayer("Name","relu_9_2_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3_1")
    maxPooling2dLayer([2 2],"Name","maxpool_2_2","Padding",[0 1 0 1],"Stride",[3 3])
    convolution2dLayer([3 1],128,"Name","conv_7_2")
    batchNormalizationLayer("Name","batchnorm_7_2")
    reluLayer("Name","relu_7_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],128,"Name","conv_8_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8_2")
    reluLayer("Name","relu_8_2")
    convolution2dLayer([3 1],128,"Name","conv_9_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9_2")
    reluLayer("Name","relu_9_2_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3_2")
    reluLayer("Name","relu_9_1")
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(100,"Name","fc_2_1","BiasInitializer","narrow-normal")
    reluLayer("Name","relu_9")
    fullyConnectedLayer(50,"Name","fc_2_2","BiasInitializer","narrow-normal")
    reluLayer("Name","relu_10")
    fullyConnectedLayer(1,"Name","fc_2_3","BiasInitializer","narrow-normal")
    regressionLayer("Name","routput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lgraph = connectLayers(lgraph,"relu_1","conv_2");
lgraph = connectLayers(lgraph,"relu_1","addition_1/in2");
lgraph = connectLayers(lgraph,"relu_3","addition_1/in1");
lgraph = connectLayers(lgraph,"relu_4","conv_5");
lgraph = connectLayers(lgraph,"relu_4","addition_2/in2");
lgraph = connectLayers(lgraph,"relu_6","addition_2/in1");
lgraph = connectLayers(lgraph,"relu_7_1","conv_8_1");
lgraph = connectLayers(lgraph,"relu_7_1","addition_3_1/in2");
lgraph = connectLayers(lgraph,"relu_9_2_1","addition_3_1/in1");
lgraph = connectLayers(lgraph,"relu_7_2","conv_8_2");
lgraph = connectLayers(lgraph,"relu_7_2","addition_3_2/in2");
lgraph = connectLayers(lgraph,"relu_9_2_2","addition_3_2/in1");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


options = trainingOptions('adam', ...
    'MiniBatchSize',128, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',true,...
    'MaxEpochs',100,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.5,...
    'LearnRateDropPeriod',5 ... 
    );%'InitialLearnRate',0.001);

net = trainNetwork(XTrain,YTrain,lgraph,options);

[YtrainPred] = predict(net,XTrain);

[YValPred] = predict(net,XValidation);

error = YValPred - YValidation;

%sequence 



