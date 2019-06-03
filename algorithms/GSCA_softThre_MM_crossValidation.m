function [cvErrors,ranks,sigSqus,opts_inner] = GSCA_softThre_MM_crossValidation(X1,X2,fun,K,opts)

% A missing value based cross validation procedure for estimating the
% prediction error of the GSCA model. The Wold partiton strategy is used.
% K should not be proportional to the number of samples. Otherwise, the
% whole row can be missing.
%
% Input:
%      X1, X2, fun, opts: same as GSCA_softThre_MM.m
%      K: number of folds.
%
% Output:
%       cvErrors: CV errors 
%       ranks: estimated ranks
%       sigSqus: estimated \sigma^2 in fitting X2
%       opts_inner: opts used inside the CV process 

% check which link function is used
if isfield(opts, 'link'),    link   = opts.link;         else link = 'logit'; end
% obj loss function for binary data and its gradient
if strcmp(link,'logit')
    obj_binary   = str2func('obj_f_logistic');
elseif strcmp(link,'probit')
    obj_binary   = str2func('obj_f_probit');
end

% size of X1 and X2
[~,n1] = size(X1); 
[m,n2] = size(X2);
mn1 = m*n1; 
mn2 = m*n2;

% using user specified parameters opts
opts_inner = opts;

% create zero matrix to hold results
cvErrors = zeros(K,1);
ranks    = zeros(K,1);
sigSqus  = zeros(K,1);
nonConv  = 0;

% K fold EM-Wold cross validation
for k = 1:K
    % select K folds in a diagonal style
    missingPatternX1 = k:K:(mn1);
    missingPatternX2 = k:K:(mn2);
    
    % set k-th fold elements as missing data
	missingX1 = X1(missingPatternX1);
    missingX2 = X2(missingPatternX2);
    Y1 = X1; Y1(missingPatternX1) = nan;
    Y2 = X2; Y2(missingPatternX2) = nan;
    
    % adaptively change lambda according to the number of non-missing elements
    missing_number = length(missingPatternX1) + length(missingPatternX2); 
    opts_inner.lambda = (1- (missing_number/(mn1+mn2)))*opts.lambda;
    
    % using remaining data to construct a GSCA model
    [mu,Z,sigmaSquare,out_inner] = GSCA_softThre_MM(Y1,Y2,fun,opts_inner);
    
    if (out_inner.iter <= 2),
        disp('previous solutions are too good to be the initilizations; need restart');
        if isfield(opts_inner,'Z0'), 
            opts_inner = rmfield(opts_inner,'Z0');
            [mu,Z,sigmaSquare,out_inner] = GSCA_softThre_MM(Y1,Y2,fun,opts_inner);
        end
    end
    
    % warm start
    if (out_inner.convStatu == 0 || (out_inner.rank > 100)),
	    nonConv = 1;
        if isfield(opts_inner, 'Z0'),
            opts_inner = rmfield(opts_inner,'Z0');
        end
    elseif (out_inner.convStatu == 1 && (nonConv == 0)),
        opts_inner.mu0 = mu;
        opts_inner.Z0  = Z;
        opts_inner.sigmaSquare0 = sigmaSquare;
    end
    
    % extract the estimated parameters for the prediction of missing elements
    Theta  = ones(m,1)*mu' + Z;
    Theta1 = Theta(:,1:n1);
    Theta2 = Theta(:,(n1+1):end);       
    missingTheta1 = Theta1(missingPatternX1);
    missingTheta2 = Theta2(missingPatternX2);
        
    % take the negative loglikelihood as the prediction error
    predictionError = obj_binary(missingX1,missingTheta1) +...
            (1/(2*sigmaSquare))*norm(missingX2-missingTheta2,'fro')^2 ...
            + length(missingPatternX2)*0.5*log(2*pi*sigmaSquare);
    cvErrors(k,1) = predictionError/missing_number; % scaled prediction error
    ranks(k,1)    = out_inner.rank;                 % estimated ranks
    sigSqus(k,1)  = sigmaSquare;                    % estimated sigma Square
end

end

