function [J, grad] = cofiCostFuncNN(params, Y, R, num_users, num_movies, ...
                                  num_features, labels,  lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%
nm=num_movie;
nu=num_user;
n=num_featuers;
L=labels; % number of labels

%Forward propagation
% -------------------------------------------------------------

X=eye(nm); % The i-th movie is represented by am 1-by-nm vector 
%whose i-th element is one and the rest are zeros 

a1=[ones(m, 1), X]; % adding the bias unity column 
% a1 size= (nm, nm+1)

z2=a1*Theta1';
% Theta1 size= (n, nm+1)
% a1 size= (nm, nm+1)
% z2 size= (nm, n)

a2=sigmoid(z2);
% a2 size= (nm, n)


a2=[ones(nm, 1), a2]; % Adding the bias unity column
%a2 size= (nm, n+1)

z3=a2*Theta2';
% Theta2 size= (nu, n+1)
%a2 size= (nm, n+1)
%z3 size= (nm, nu)


a3=sigmoid(z3);
%a3 size= (nm, nu)


a3=[ones(nm, 1), a3]; % Adding the bias unity column
%a2 size= (nm, nu+1)


z4=a3*Theta3';
% Theta3 size= (L, nu+1)
% a3 size= (nm, nu+1)
%z3 size= (nm, L)

a4=sigmoid(z4);
%a4 size= (nm, L)


h=a4s;
%z3 size= (nm, L)


for l=1:L
    
    yk=(y==k);
    % yk size = (m, 1)
    Jk=-1/m * (yk'*log(h(:,k))+(1-yk')*log(1-h(:,k)));
    J=J+Jk;
end

reg=lambda/2/m*(sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2))); % Regularization term

J=J+reg; % Cost function

