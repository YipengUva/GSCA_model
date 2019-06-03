function [ out ] = obj_f_probit( X1, Z, sigma)
%object function of probit loss
% X1: binary data matrix
% Z: offset + AB', the log-odds
% Z = ones(m,1)*mu + A*B1';

% probit loss for an individual binary element 
%   if x = 1, the probit loss is -log(cdf(z))
%   if x = 0, the probit loss is -log(1-cdf(z))

if(nargin<3)
    sigma = 1;
end

[m,n]    = size(X1);
loss_mat = NaN(m,n);
one_ind  = (X1==1); % index for xij == 1 
zero_ind = (X1==0); % index for xij == 0
loss_mat(one_ind)  = -log(gausscdf(Z(one_ind),0,sigma));
loss_mat(zero_ind) = -log(1-gausscdf(Z(zero_ind),0,sigma));
out = sum(sum(loss_mat));
end