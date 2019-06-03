function [RMSE_mat,RV_mat,sigSqus,ranks,RMSE_ind] = ...
    GSCA_softThre_MM_modelSelection_RMSEs_simulation(X1,X2,Theta_simu,mu_simu,Z_simu,fun,lambdas,opts)

% When simulated data sets are used, the simulated Theta, mu, Z are
% available. This function is used to compute the respect RMSEs, RV
% coefficients in estimating mu, Z and Theta from the GSCA model for a
% sequence of lambdas.
%
% Input:
%      X1, X2, fun, opts: same as GSCA_softThre_MM.m
%      Theta_simu: simulated Theta 
%      mu_simu: simulated mu 
%      Z_simu: simulated Z 
%
% Output:
%       RMSE_mat: length(lambdas)*3; columns indicates RMSEs in estimating Theta, mu, Z 
%       RV_mat: length(lambdas)*3;   columns indicates RV coefficients in estimating Theta, mu, Z 
%       ranks: length(lambdas)*1;    estimated ranks
%       sigSqus: length(lambdas)*1;  estimated \sigma^2 in fitting X2
%       RMSE_ind: length(lambdas)*4; columns indicates RMSEs in estimating Theta_1, Theta_2, Z_1, Z_2

	
% structure to hold results
[m,n1] = size(X1);
num_iter = length(lambdas);
RMSE_mat = nan(num_iter,3);
RV_mat   = nan(num_iter,3);
sigSqus  = nan(num_iter,1);
ranks    = nan(num_iter,1);
RMSE_ind = nan(num_iter,4);

% iterations
for j=1:num_iter
    lambda = lambdas(j);
    opts.lambda = lambda;
    [mu,Z,sigSqu,out] = GSCA_softThre_MM(X1,X2,fun,opts);
    ThetaHat = ones(m,1)*mu' + Z;
	
	% RMSEs and RVs in eastimating Theta, mu, Z
    RMSE_Theta = norm(Theta_simu-ThetaHat,'fro')^2/norm(Theta_simu,'fro')^2;
	RMSE_mu    = norm(mu_simu-mu,'fro')^2/norm(mu_simu,'fro')^2;
	RMSE_Z     = norm(Z_simu-Z,'fro')^2/norm(Z_simu,'fro')^2;
	RV_Theta   = RV_modified_bda(Theta_simu, ThetaHat);
	RV_mu      = RV_modified_bda(mu_simu, mu);
	RV_Z       = RV_modified_bda(Z_simu, Z);
	
	RMSE_mat(j,:) = [RMSE_Theta,RMSE_mu,RMSE_Z];
    RV_mat(j,:)   = [RV_Theta,RV_mu,RV_Z];
	
	% estimated ranks and \sigma^2
	sigSqus(j,1)  = sigSqu;
    ranks(j,1)    = out.rank;
	
	% RMSEs in estimating Theta_1, Theta_2, Z_1, Z_2
    RMSE_Theta1 = norm(Theta_simu(:,1:n1)-ThetaHat(:,1:n1),'fro')^2/norm(Theta_simu(:,1:n1),'fro')^2;
    RMSE_Theta2 = norm(Theta_simu(:,(n1+1):end)-ThetaHat(:,(n1+1):end),'fro')^2/norm(Theta_simu(:,(n1+1):end),'fro')^2;
    RMSE_Z1     = norm(Z_simu(:,1:n1)-Z(:,1:n1),'fro')^2/norm(Z_simu(:,1:n1),'fro')^2;
    RMSE_Z2     = norm(Z_simu(:,(n1+1):end)-Z(:,(n1+1):end),'fro')^2/norm(Z_simu(:,(n1+1):end),'fro')^2;
	
    RMSE_ind(j,:) = [RMSE_Theta1, RMSE_Theta2, RMSE_Z1, RMSE_Z2];
	
end

end