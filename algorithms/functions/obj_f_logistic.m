function [ out ] = obj_f_logistic( X1, Z )
% object function of logistic loss
% X1: binary data matrix
% Z: offset + AB', the log-odds
% Z = ones(m,1)*mu + A*B1';
% when x=1, the loss is -log(fai(z)), equal -log(1/(1+exp(-z))).
% when x=0, the loss is -log(1-fai(z)), equal -log(1/(1+exp(z)).
% The following is the matrix form
X = 2*X1 - 1;
temp = 1./(1+exp(-X.*Z));
out = -sum(sum(log(temp),'omitnan'));
end

