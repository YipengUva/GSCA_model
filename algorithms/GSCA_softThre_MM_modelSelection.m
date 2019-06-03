function [cvErrors_mat,sigSqus_mat,ranks_mat] = GSCA_softThre_MM_modelSelection(X1,X2,fun,K,lambdas,opts)

% The model selection procedure. For a sequence of lambdas, apply the CV
% procedure to the GSCA models with every lambda.
%
% Input:
%      X1, X2, fun, opts, K: same as GSCA_softThre_MM_crossValidation.m
%      lambdas: a sequence of values of lambda
%
% Output:
%       cvErrors_mat: a length(lambdas)*K matrix to hold the CV errors 
%       ranks: a length(lambdas)*K matrix to hold the estimated ranks
%       sigSqus: a length(lambdas)*K matrix to hold the estimated \sigma^2 in fitting X2

% structure to hold results
num_iter     = length(lambdas);
cvErrors_mat = nan(num_iter,K);
sigSqus_mat  = nan(num_iter,K);
ranks_mat    = nan(num_iter,K);

% initilize lambda to lambda0
for j = 1:num_iter
    lambda = lambdas(j);
	opts.lambda = lambda;
    [cvErrors,ranks,sigSqus,~] = GSCA_softThre_MM_crossValidation(X1,X2,fun,K,opts);
    cvErrors_mat(j,:) = cvErrors';
    ranks_mat(j,:)    = ranks';
    sigSqus_mat(j,:)  = sigSqus';
end

end