function [gradients, loss, dlYPred] = modelGradients(dlX, adjacencyTrain, T, parameters)

dlYPred = model(dlX, adjacencyTrain, parameters);

loss = crossentropy(dlYPred, T, 'DataFormat', 'BC');

gradients = dlgradient(loss, parameters);

end