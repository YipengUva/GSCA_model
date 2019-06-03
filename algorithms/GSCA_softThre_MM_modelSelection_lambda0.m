function [lambda] = GSCA_softThre_MM_modelSelection_lambda0(X1,X2,fun,opts)

% using low precision models to find a proper value of lambda0

% use a low precision model
opts.tol_obj = 1e-2;
maxiteration = 1000;

% initialization
lambda0 = 100; 
       
for i = 1:maxiteration
    lambda = lambda0;
    opts.lambda = lambda;
    [~,~,~,out] = GSCA_softThre_MM(X1,X2,fun,opts);
    if (out.rank <= 1)
        break;
    elseif (out.rank > 1)
        lambda = lambda/(0.9);
    end
    lambda0 = lambda;
end
lambda = lambda0;
end
