function [X1,X2,Theta_simu,mu_simu,Z_simu,E_simu] = GSCA_data_simulation(mu1_fixed,SNRs,K,link,seed)

% The simulation of coupled binary and quantitative data sets. In this
% example the sample size is fixed to 160; the number of binary and
% quantitative variables are 410 and 1000 respectively. The noise level in
% simulating quantitative X2 is fset to 1.
%
% Input:
%      mu1_fixed: pre-defined offset term for binary data
%      SNRs: 2*1 vector; SNRs = [SNR1, SNR2]; pre-defined SNRs 
%      K: simulated low rank
%      link: link function used
%      seed: set seed to reproduce the simulation
%
% Output:
%       X1: simulated binary data X1
%       X2: simulated quantitative data X2
%       Theta_simu: simulated Theta
%       mu_simu: simulated mu
%       Z_simu: simulated Z 
%       E_simu: simulated noise term E

% fixed parameters
m  = 160;
n1 = 410;
n2 = 1000;
noise_X2 = 1; % noise level in simulating X2

% pre-defined parameters
SNR1 = SNRs(1);
SNR2 = SNRs(2);
if(nargin<4), link = 'logit'; end
if(exist('seed')), rng(seed); end % set seed to reproduce the results

% simulated diagonal matrix D
D = diag(sort(abs(randn(K,1)),'descend'));

% offset term
mu1_simu = mu1_fixed;
mu2_simu = randn(1,n2); 

% simulation of common score matrix A, and distinct loading matrices B1 and B2
[A_simu,~]  = qr(mvnrnd(zeros(1,K),eye(K),m),0);
[B1_simu,~] = qr(mvnrnd(zeros(1,K),eye(K),n1),0);
[B2_simu,~] = qr(mvnrnd(zeros(1,K),eye(K),n2),0);

% simulation of error matrices E1 and E2
E2_simu   = sqrt(noise_X2)*randn(m,n2);

if strcmp(link,'logit')
    SLogistic = makedist('Logistic');
	E1_simu   = random(SLogistic, m,n1);
elseif strcmp(link,'probit')
    E1_simu   = randn(m,n1);
end

% compute c1 and c2 to satisfy the pre-defined SNRs
Z1_pre = A_simu*D*B1_simu';
Z2_pre = A_simu*D*B2_simu';
c1 = sqrt(SNR1)*norm(E1_simu ,'fro')/norm(Z1_pre,'fro');
c2 = sqrt(SNR2)*norm(E2_simu ,'fro')/norm(Z2_pre,'fro');
Z1_simu = c1*Z1_pre;
Z2_simu = c2*Z2_pre;

% simulate Theta1 Theta2
Theta1_simu = ones(m,1)*mu1_simu + Z1_simu;
Theta2_simu = ones(m,1)*mu2_simu + Z2_simu;

% generating binary data matrix X1
if strcmp(link,'logit')
    fM_simu = fai_logistic(Theta1_simu);
elseif strcmp(link,'probit')
    fM_simu = gausscdf(Theta1_simu,0,1);
end
Y  = sign(fM_simu - rand(m,n1)); 
X1 = (0.5).*(Y+1);

% generating binary continuous matrix X2
X2 = Theta2_simu + E2_simu;

% test if simulated X1 have columns with all of 0s
nonZeroCol  = (sum(X1,1) > 0);
X1 = X1(:,nonZeroCol);
mu1_simu    = mu1_simu(nonZeroCol);
Z1_simu     = Z1_simu(:,nonZeroCol);
E1_simu     = E1_simu(:,nonZeroCol);
Theta1_simu = Theta1_simu(:,nonZeroCol);

% outputs
mu_simu = [mu1_simu'; mu2_simu'];
Z_simu  = [Z1_simu, Z2_simu];
E_simu  = [E1_simu, E2_simu];
Theta_simu = [Theta1_simu, Theta2_simu];

% deflate the column offset of Z_simu
mu_simu = mu_simu + mean(Z_simu,1)';
Z_simu  = Z_simu - ones(m,1)*mean(Z_simu,1);

end