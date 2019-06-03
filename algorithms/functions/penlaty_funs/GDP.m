function y = GDP(x,gamma,lambda)
% GDP penalty for singular values
x = abs(x) ;
y = lambda*log(1+x./gamma);

