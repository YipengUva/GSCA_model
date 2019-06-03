function [ out ] = KL_divergence(P,Q)
% function to calculate the KL divergence between two probability matrices P and Q
%   element wise 
[m,n] = size(P);
KL_matrix = P.*log(P./Q) + (1-P).*log((1-P)./(1-Q));
out   = sum(sum(KL_matrix))/(m*n);
end