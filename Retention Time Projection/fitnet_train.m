function [Y] = fitnet_train(inputs,outputs)
    
	
net = feedforwardnet(25);   %隐藏层为10层
	
% Train the Network
[net,~] = train(net,inputs,outputs,'useParallel','yes');

% Test the Network
Y = net(inputs);

save feedforward.mat net

Y = Y';

end