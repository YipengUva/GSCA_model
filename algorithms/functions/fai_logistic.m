function [ out ] = fai_logistic( x )
%logistic function
%   element wise logistic function
out = 1./(1+exp(-x));

end

