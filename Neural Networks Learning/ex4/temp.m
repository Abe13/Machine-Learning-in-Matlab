% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%=============================================================

a1=[ones(m, 1), X]; % adding the unity column 

z2=a1*Theta1'; 
% Theta1 size= (hidden_layer_size, (input_layer_size + 1))
%a1 size = (m, (input_layer_size+1));
%z2 size= (m, hidden_layer_size)

a2=sigmoid(z2);
%a2 size= (m, hidden_layer_size)

a2=[ones(m, 1), a2];
%a2 size= (m, hidden_layer_size+1)

z3=a2*Theta2';
% Theta2 size= (num_labels, hidden_layer_size + 1)
%a2 size= (m, hidden_layer_size+1)
%z3 size= (m, num_labels)


a3=sigmoid(z3);
%a3 size= (m, num_labels)
h=a3;



for k=1:num_labels
    
    yk=(y==k);
    % yk size = (m, 1)
    
    Jk=-1/m * (yk'*log(h(:,k))+(1-yk')*log(h(:,k)));
    J=J+Jk;
end

lambda=0;


reg=lambda/2/m*(sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2))); % Regularization term

J=J+reg; % Cost function
    
