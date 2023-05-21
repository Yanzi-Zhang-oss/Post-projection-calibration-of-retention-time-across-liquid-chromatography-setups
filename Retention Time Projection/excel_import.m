%% 导入电子表格中的数据
% 用于从以下电子表格导入数据的脚本:
%
%    工作簿: C:\Users\Administrator\Desktop\RT_prediction\2021104-整合简单模式给崔文超-相同仪器-不同色谱条件-RT移植-Vanquish QE PLus(1).xlsx
%    工作表: 内部训练和测试集
%
% 由 MATLAB 于 2021-11-06 14:22:19 自动生成

%% 设置导入选项并导入数据

function [RTVanquishQEPLus1] = excel_import()
opts = spreadsheetImportOptions("NumVariables", 14);

% 指定工作表和范围
opts.Sheet = "内部训练和测试集";
opts.DataRange = "M7:Z261";

% 指定列名称和类型
opts.VariableNames = ["RSLC1", "GOLD2", "T31", "GOLD4", "RSLC4", "RSLC5", "RSLC3", "RSLC7", "RSLC8", "GOLD1", "GOLD3", "RSLC2", "GOLD5", "RSLC6"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% 导入数据
RTVanquishQEPLus1 = readtable("C:\Users\Administrator\Desktop\RT_prediction\2021104-整合简单模式给崔文超-相同仪器-不同色谱条件-RT移植-Vanquish QE PLus(1).xlsx", opts, "UseExcel", false);

%% 转换为输出类型
RTVanquishQEPLus1 = table2array(RTVanquishQEPLus1);

%% 清除临时变量
clear opts

end