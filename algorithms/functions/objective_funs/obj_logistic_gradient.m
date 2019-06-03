function [ out ] = obj_logistic_gradient(X,Theta)
% gradient of logistic loss
% X: binary data matrix
% Theta: offset + Z, the log-odds
% The following is the matrix form
out = fai_logistic(Theta)-X;
end

