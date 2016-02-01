clc
clear

load ex3data1.mat
load ex3weights.mat


m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



% =========================================================================

    
X=[ones(m,1) X]; % Adding one to the matrix
a1=X;

z2=a1*Theta1';
a2=1./(1+exp(z2));

m2=size(a2,1);

a2=[ones(m2,1) a2 ]; %Augmented a2 matrix, adding column 1

z3=a2*Theta2';
a3=1./(1+exp(z3));

[C, p]=max(a3');

p=p';
