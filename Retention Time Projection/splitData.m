function [train,validation,test] = splitData(RT,symbol,...
                                ratio_train,ratio_validation,ratio_test)

train = [];
validation = [];
test = [];

[m,~] = size(symbol);

list = cell(1,max(symbol));
for j =1:1:max(symbol)
    for i = 1:1:m
        if symbol(i) == j
            list{j} = [list{j};RT(i,:)];
        end
    end
end

for i = 1:1:max(symbol)
    numMolecules = size(list{i}, 1);
    
    % Set initial random state for example reproducibility.
    rng(0);
    
    if size(list{i}, 1) ~= 0
        % Get training data
        idx = randperm(size(list{i}, 1), floor(ratio_train*numMolecules));
        temp_train = list{i}(idx,:);
        list{i}(idx,:) = [];
        
        % Get validation data
        idx = randperm(size(list{i}, 1), floor(ratio_validation*numMolecules));
        temp_validation = list{i}(idx,:);
        list{i}(idx,:) = [];
        
        % Get test data
        temp_test = list{i};
        
        train = [train;temp_train];
        validation = [validation;temp_validation];
        test = [test;temp_test];
    end
end

end