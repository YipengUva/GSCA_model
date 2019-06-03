function y = GDP_sg(x,gamma,lambda)
%  supergradient of GDP
x = abs(x) ;
y = lambda./(gamma + x);
