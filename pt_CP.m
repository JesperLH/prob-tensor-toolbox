function [E_FACT,E_FACT2, E_Lambda, Lowerbound, model,all_samples]=pt_CP(X,D,constraints,varargin)


    [E_FACT,E_FACT2, E_Lambda, Lowerbound, model,all_samples]=VB_CP_ALS(X,D,constraints,varargin{:});

end