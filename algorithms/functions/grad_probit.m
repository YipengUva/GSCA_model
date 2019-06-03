function [ out ] = grad_probit( X1, Z, sigma )
% calculate the gradient for each ijth element of z in matrix form
%   if xij = 1, the gradient is -pdf(zij)/cdf(zij)
%   if xij = 0, the radient is pdf(zij)/(1-cdf(zij))
[m,n1] = size(X1);  % size of X1
out    = NaN(m,n1); % str to hold gradient
one_ind  = (X1==1); % index for xij == 1 
zero_ind = (X1==0); % index for xij == 0
out(one_ind) = -gausspdf(Z(one_ind),0,sigma)./gausscdf(Z(one_ind),0,sigma);
out(zero_ind) = gausspdf(Z(zero_ind),0,sigma)./(1-gausscdf(Z(zero_ind),0,sigma));
end

