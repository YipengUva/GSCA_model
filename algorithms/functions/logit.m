function [ out ] = logit( x )
% logit function
%   element wise logit function
out = log(x./(1-x));

end

