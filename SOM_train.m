function [output] = SOM_train(MIAOSHUFU,GROUP_NUM)


inputs = MIAOSHUFU';
[inputs] = normalize_fangcha(inputs);  %方差回归

% Create a Self-Organizing Map
dimension1 = GROUP_NUM;
dimension2 = GROUP_NUM;
net = selforgmap([dimension1 dimension2]);

% Train the Network
[net,~] = train(net,inputs);

% Test the Network
Y = net(inputs);

% View the Network
% view(net)

Y = Y';
[m,n] = size(Y);

output = zeros(m,1);

for i = 1:1:m
   for j = 1:1:n
      if Y(i,j) == 1
          output(i,1) = j;
      end
   end
end


save SOM_ANN.mat net

end
