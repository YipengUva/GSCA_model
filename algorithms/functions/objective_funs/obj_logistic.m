function [ out ] = obj_logistic(X,Theta,P)
% object function of logistic loss
% X: binary data matrix
% Theta: offset + Z, the log-odds
% Theta = ones(m,1)*mu + Z;
% P: weighting matrix 
% when x=1, the loss is -log(fai(z)), equal -log(1/(1+exp(-z))).
% when x=0, the loss is -log(1-fai(z)), equal -log(1/(1+exp(z)).
% The following is the matrix form
X(logical(1-P)) = nan; 
X = 2*X - 1;
temp = 1./(1+exp(-X.*Theta));
out = -sum(sum(log(temp),'omitnan'));
end

