function [RV]=RV_modified_bda(X,Y)
%RV modifed coefficient by bda group
AA=X*X';
BB=Y*Y';
AA0 = AA - diag(diag(AA),0);
BB0 = BB - diag(diag(BB),0);
RV = trace(AA0*BB0)/norm(AA0,'fro')/norm(BB0,'fro');
end
