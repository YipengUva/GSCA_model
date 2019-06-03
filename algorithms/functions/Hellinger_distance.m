function [ out ] = Hellinger_distance(P,Q)
% function to calculate the Hellinger_distance between two probability matrices P and Q
%   element wise 
[m,n] = size(P);
HD_matrix = (1/sqrt(2))*sqrt((sqrt(P) - sqrt(Q)).^2 + (sqrt(1-P) - sqrt(1-Q)).^2);
out = sum(sum(HD_matrix))/(m*n);

end

