function [lambdat] = GSCA_softThre_MM_modelSelection_lambdat(X1,X2,fun,lambda0,opts)

% using low precision models to find a proper value of lambdat.
% lambda0: selected lambda0 from the previous step

% use a low precision model
opts.tol_obj = 1e-2;
maxiteration = 1000;

% initialization
lambda0 = 0.5*lambda0;

for i = 1:maxiteration
    lambda = lambda0;
    opts.lambda = lambda;
    [~,~,~,out] = GSCA_softThre_MM(X1,X2,fun,opts);
    
    if (out.convStatu == 0 || out.rank > 20)
        break; % select lambdat, for which estimated rank is larger than 20
    else
        lambda = 0.9*lambda;
    end
    lambda0 = lambda;  
end
lambdat = lambda;
end
