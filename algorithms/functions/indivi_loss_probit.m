function [ out ] = indivi_loss_probit( x, z, sigma )
% probit loss for an individual binary element 
%   if x = 1, the probit loss is -log(cdf(z))
%   if x = 0, the probit loss is -log(1-cdf(z))
if(x==1)
    out = -log(gausscdf(z,0,sigma));
else
    out = -log(1-gausscdf(z,0,sigma));
end

