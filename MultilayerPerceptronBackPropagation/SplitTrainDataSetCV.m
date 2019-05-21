%Divisão do dataset de treino em 80% estimation e 20% validation

function [est, val] = SplitTrainDataSetCV(data)

    n = size(data,1);
    data_rand = data(randperm(n),:);
    m = ceil(n/5);
    k = 1:m:n-m;
    val = data_rand(k:k+m-1,:);
    est = [data_rand(1:k-1,:); data_rand(k+m:end,:)];
    
end
