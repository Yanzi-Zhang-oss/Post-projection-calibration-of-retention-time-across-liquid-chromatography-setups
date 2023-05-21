function [out] = normalize_fangcha(in)
[m,n] = size(in);
%%%一倍方差标准化
for i = 1:1:n
    x_average = mean(in(:,i));
    var_out = var(in(:,i));
    for j = 1:1:m
        in(j,i) = (in(j,i) - x_average)/var_out;
    end
end
out = in;
end