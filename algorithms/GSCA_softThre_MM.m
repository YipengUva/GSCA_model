function [mu,Z,sigmaSquare,out] = GSCA_softThre_MM(X1,X2,fun,opts)

% GSCA model of binary data set X1 and quantitative data set X2. The low
% rank constraint is surrogated by a concave function on the singular
% values. A MM algorithm is used to fit the model.
%     minimize -logP(X1|Theta1) - logP(X2|Theta2) + lambda*g(Z); 
%   s.t. Theta1 = 1*mu1' + Z1; 
%        Theta2 = 1*mu2' + Z2; 
%             Z = [Z1, Z2];
%
% Input:
%      X1: binary matrix, 0 or 1
%      X2: quantitative matrix 
%      fun: concave surrogate function of low rank constraint 
%      opts.
%           tol_obj: tolerance for relative change of f, default:1e-5
%           maxit: max number of iterations, default: 1000
%           lambda: tuning parameter of penalty g(Z)
%           gamma:  tuning parameter of penalty g(Z)
%           link: 'logit' or 'probit', default: 'logit'
%
% Output:
%       mu: offset term 
%       Z: underlying low rank matrix
%       sigmaSquare: \sigma^2 in fitting X2
%       out. 
%           iter: number of iterations
%           hist_obj: objective value at each iteration
%           f1_obj: objective value for f1 loss at each iteration
%           f2_obj: objective value for f2 loss at each iteration
%           rel_obj: relative change of obj at each iteration
%           rel_Theta: relative change of Theta at each iteration
%           convStatu: convergence status. 1: convergence; 0: non convergence
%           rank: estimated rank
%           L: estimated lipschitz constant in each iteration

% parameters and defaults
if isfield(opts, 'tol_obj'), tol_obj = opts.tol_obj;     else tol_obj = 1e-5; end
if isfield(opts, 'maxit'),   maxit  = opts.maxit;        else maxit   = 1000; end
if isfield(opts, 'lambda'),  lambda = opts.lambda;       else lambda  = 1;    end
if isfield(opts, 'gamma'),   gamma  = opts.gamma;        else gamma   = 1;    end
if isfield(opts, 'link'),    link   = opts.link;         else link = 'logit'; end

% set parameters according to the specific problem
% penalty function and its super gradient
hfun    = str2func(fun);         % penalty function 
hfun_sg = str2func([fun '_sg']); % super gradient of penalty function

% obj loss function for binary data and its gradient
if strcmp(link,'logit')
    obj_binary   = str2func('obj_logistic');
    obj_binary_g = str2func('obj_logistic_gradient');
	L0 = 0.25;
elseif strcmp(link,'probit')
    obj_binary   = str2func('obj_probit');
    obj_binary_g = str2func('obj_probit_gradient');
	L0 = 1;
end

% parameters
[~,n1] = size(X1); [m,n2] = size(X2); n = n1+n2; % size of data sets
P1 = 1-isnan(X1); P2 = 1-isnan(X2); P = [P1,P2]; % weighting matrices
X1(isnan(X1)) = 0; X2(isnan(X2)) = 0; % remove nan values
rankW2 = sum(sum(P2));  % the number of nonzero elements in X2

% initialization
if(isfield(opts, 'Z0')) % using imputed initialization
    mu0 = opts.mu0; mu0t = mu0'; mu10t = mu0t(1:n1); mu20t = mu0t((n1+1):end);
	Z0  = opts.Z0; Z10  = Z0(:,1:n1);  Z20 = Z0(:,(n1+1):end);
    if(isfield(opts,'sigmaSquare0')), sigmaSquare0 = opts.sigmaSquare0; 
    else sigmaSquare0 = 1; end
else                    % using random initialization
    mu0t = zeros(1,n); mu10t = mu0t(1:n1); mu20t = mu0t((n1+1):end);
    Z0  = rand(m,n);   Z10  = Z0(:,1:n1);  Z20 = Z0(:,(n1+1):end);
    sigmaSquare0 = 1;   % using \sigma^2 = 1 as initialization
end

sigma0  = svd(Z0,'econ'); % singular value of Z0
Theta10 = ones(m,1)*mu10t + Z10;
Theta20 = ones(m,1)*mu20t + Z20; 
Theta0  = [Theta10, Theta20];

% initial value of loss function
f1_obj0 = obj_binary(X1,Theta10,P1); % loss function for fitting binary data X1
f2_obj0 = (1/(2*sigmaSquare0))*norm(P2.*(X2-Theta20),'fro')^2 ... 
    + 0.5*rankW2*log(sigmaSquare0) + 0.5*rankW2*log(2*pi); % loss function for X2
obj0    = f1_obj0 + f2_obj0 + sum(hfun(sigma0,gamma,lambda)); % loss function + penalty
out.f1_obj(1)   = f1_obj0;
out.f2_obj(1)   = f2_obj0;
out.hist_obj(1) = obj0;
out.sigmaSquares(1) = sigmaSquare0;

% specify the column centering matrix 
Jcentering = (eye(m) - (1/m)*ones(m,1)*ones(m,1)'); % save centering matrix 

% iterations
for k = 1:maxit
    %fprintf('%d th iteration\n',k);
    
    % cached previous results
    Theta1 = Theta10; Theta2 = Theta20; Theta  = Theta0;
    sigmaSquare = sigmaSquare0; 
	
    % forming H^k for the majorization
    gradient_f1 = obj_binary_g(X1,Theta1); 
    gradient_f2 = (1/sigmaSquare)*(Theta2 - X2);
	gradient_f  = [P1.*gradient_f1, P2.*gradient_f2];
    
    L  = max(L0, (1/sigmaSquare));
	Hk = Theta - (1/L)*gradient_f;
	
    %--- update mu1 and mu2 ---
	mut  = mean(Hk,1); % column mean of tildeHk as the mean
	mu1t = mut(1:n1); 
	mu2t = mut((n1+1):end);
	
    % centering H^k using the centering matrix J
    JH  = Jcentering*Hk;
    
    %--- update Z ---
    % SVD of JH
    [D, S, V] = svd(JH,'econ');
	
	% weighted singular value threholding
    sigma   = diag(S);                 % extract singular values
    w = hfun_sg(sigma0,gamma,lambda);  % supergradient of penality function on current estimated singular values
    if all(isinf(w))==1                % to tackle nonproper initialization
        w = hfun_sg(sigma,gamma,lambda);
    else
        w = w;
    end
            
    sigma = max(0,sigma - (w/L));  % weighted singular threhoulding	
    svp   = length(find(sigma>0)); % number of nonzero singular values
	sigma = sigma(1:svp);
	
	% update Z
	Z  = D(:,1:svp)*diag(sigma)*V(:,1:svp)';
	Z1 = Z(:,1:n1); 
	Z2 = Z(:,(n1+1):end);
	
    Theta1 = ones(m,1)*mu1t + Z1;
	Theta2 = ones(m,1)*mu2t + Z2;
	Theta  = [Theta1, Theta2];
	
	%---- update sigmaSquare ----
	sigmaSquare = (norm(P2.*(X2 - Theta2), 'fro'))^2/(rankW2);
    
    if (sigmaSquare<0.05) % return to previous estimations
        out.sigSqu_extre = sigmaSquare;
        mut         = mu0t;
        sigmaSquare = sigmaSquare0;
        Z           = Z0;
        svp         = length(find(sigma0>0));
        k           = k-1;
        break;
    end
        
    % diagnostics
    f1_obj = obj_binary(X1,Theta1,P1); 
    f2_obj = (1/(2*sigmaSquare))*norm(P2.*(X2-Theta2),'fro')^2 ...
        + 0.5*rankW2*log(sigmaSquare) + 0.5*rankW2*log(2*pi);
    obj   = f1_obj + f2_obj + sum(hfun(sigma,gamma,lambda)); 

    % reporting
    out.hist_obj(k+1) = obj;
    out.f1_obj(k+1)   = f1_obj;
    out.f2_obj(k+1)   = f2_obj;
    
	out.rel_obj(k)    = (obj0-obj)/(obj0);
	out.rel_Theta(k)  = norm(Theta0-Theta,'fro')^2/norm(Theta0,'fro')^2;
    out.sigmaSquares(k)= sigmaSquare;
    out.L(k)          = L;
    
    % stopping checks
    if (out.rel_obj(k) < tol_obj); break; end;
    
    % save previous results
    Theta10 = Theta1; Theta20  = Theta2; Theta0 = Theta; L0 = L;
	obj0   = obj; sigmaSquare0 = sigmaSquare;
    sigma0 = [sigma;zeros(min(m,min(n1,n2))-svp,1)];
    mu0t   = mut;
end

 if (isfield(out, 'sigSqu_extre')) 
    fprintf('\n warning: non convergence; penalty is not strong enougth;\n non low rank structure is found \n'); 
	out.convStatu = 0;
 else 
	out.convStatu = 1;
 end;
 
 out.iter = k;
 out.rank = svp;
 mu = mut';
 
end



