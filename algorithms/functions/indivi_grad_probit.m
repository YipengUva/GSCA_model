function [ out ] = indivi_grad_probit( x, z, sigma )
% calculate the gradient for each ijth element of z
%   if x = 1, the gradient is -pdf(z)/cdf(z)
%   if x = 0, the radient is pdf(z)/(1-cdf(z))
if(x==1)
    out = -gausspdf(z,0,sigma)/gausscdf(z,0,sigma);
else
    out = gausspdf(z,0,sigma)/(1-gausscdf(z,0,sigma));
end

