# Demos and Matlab functions for the GSCA models of binary and quantitative data sets
Author: Yipeng Song

Research group: Biosystems Data Analysis Group, Uiversity of Amsterdam

Make sure the current folder is in your Matlab path and run the help function from the Matlab console (e.g. help GSCA_softThre_MM) for more information on the input/output of the algorithms.

## Demos 

The demos are used to show how to simulate binary and quantitative data sets (low signal-to-noise ratio and imbalanced binary data), how to construct a GSCA model, and how to do model selection. The docs for and the results of these two demos can be found in the examples folder.
 
demo_GSCA_model.m:
    A demo to show how to simulate coupled binary and quantitative data
	sets, and how to fit a GSCA model. 
	
demo_GSCA_model_selection.m:
    A demo to show how to do model selection of the GSCA model based on 
	simulated data sets.
	
examples:

    demo_GSCA_model.html: 
	    Documentation for running demo_GSCA_model.m
    demo_GSCA_model_selection.html:	
	    Documentation for running demo_GSCA_model_selection.m

## Algorithms
 
The algorithms used to do data simulation, to fit a GSCA model, to do model selection are in the algorithms folder.  
		
Algorithms:

    X1_CNA.mat:
	    An example of imbalanced binary data set. The logit transform of the empirical 
		marginal probabilities are used as the offset term in simulation.
		
    GSCA_data_simulation.m:
        A function used to simulate binary and quantitative data sets according to the GSCA model with logit or probit link.
		
    GSCA_hardThre_MM.m:
        The algorithm to fit a GSCA model with exact low rank constraint.
		
	GSCA_softThre_MM.m:
        The algorithm to fit a GSCA model with $L_q$, SCAD, and GDP penalties. Nuclear norm penalty is included as a special case of $L_q$ penalty.	
		
	GSCA_softThre_MM_crossValidation.m:
        A missing value based K-folder CV process.	
		
    GSCA_softThre_MM_modelSelection.m:
        Model selection process based on the above CV procedure.
		
	GSCA_softThre_MM_modelSelection_lambda0.m:
        To select a $\lambda_0$, which is large enough to achieve at most rank 1 estimation.
		
	GSCA_softThre_MM_modelSelection_lambdat.m:
        To select a $\lambda_t$, which is small enough to achieve rank 20 estimation.
		
    GSCA_softThre_MM_modelSelection_RMSEs_simulation.m: 
	    When simulated data sets are used, how to compute the RMSEs in estimating the underlying structures for multiple models.	
		
    functions:
        Functions used in the algorithms.


