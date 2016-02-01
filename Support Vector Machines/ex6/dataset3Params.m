function [C_max, sigma_max] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


Cv=0.2:0.01:1;
sigmav=0.05:0.01:0.3;

[Cg,sigmag]=meshgrid(Cv, sigmav);

Pr=zeros(size(Cg)); % Error matrix

Pr_max=0;
C_max=C;
sigma_max=sigma;

for i=1:length(sigmav)
     
    for j=1:length(Cv)
        sigma=sigmav(i);
        C=Cv(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        ypred=svmPredict(model, Xval);
        
        Pr(i,j)=mean(ypred==yval);
        if Pr(i,j)>Pr_max
            Pr_max=Pr(i,j);
            C_max=C;
            sigma_max=sigma;
        end
    end
end






% =========================================================================

end
