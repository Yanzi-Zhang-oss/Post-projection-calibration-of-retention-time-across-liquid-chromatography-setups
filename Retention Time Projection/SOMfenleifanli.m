[RT] = excel_import();

[symbol] = SOM_train(RT,2);

ratio_train = 0.5;
ratio_validation = 0.4;
ratio_test = 0.1;

[train,validation,test] = splitData(RT,symbol,...
                                ratio_train,ratio_validation,ratio_test);