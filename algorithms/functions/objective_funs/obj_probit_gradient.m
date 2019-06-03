function [ out ] = obj_probit_gradient( X, Theta)
% calculate the gradient for each ijth element of Theta in matrix form
%   if xij = 1, the gradient is -pdf(thetaij)/cdf(thetaij)
%   if xij = 0, the radient is pdf(thetaij)/(1-cdf(thetaij))
[m,n]  = size(X);  % size of X
out    = NaN(m,n); % structure to hold gradient
one_ind  = (X==1); % index for xij == 1 
zero_ind = (X==0); % index for xij == 0
out(one_ind) = -gausspdf(Theta(one_ind),0,1)./gausscdf(Theta(one_ind),0,1);
out(zero_ind) = gausspdf(Theta(zero_ind),0,1)./(1-gausscdf(Theta(zero_ind),0,1));
end

