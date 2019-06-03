function [ out ] = obj_probit( X, Theta, P)
%object function of probit loss
% X: binary data matrix
% Theta: offset + AB', the log-odds
% Theta = ones(m,1)*mu + A*B1';
% P: weighting matrix 
% probit loss for an individual binary element 
%   if x = 1, the probit loss is -log(cdf(z))
%   if x = 0, the probit loss is -log(1-cdf(z))
[m,n]    = size(X);
loss_mat = NaN(m,n);
one_ind  = (X==1); % index for xij == 1 
zero_ind = (X==0); % index for xij == 0
loss_mat(one_ind)  = -log(gausscdf(Theta(one_ind),0,1));
loss_mat(zero_ind) = -log(1-gausscdf(Theta(zero_ind),0,1));
out = sum(sum(loss_mat(logical(P))));
end