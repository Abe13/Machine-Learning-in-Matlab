%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     estimateGaussian.m
%     selectThreshold.m
%     cofiCostFunc.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% =============== Part 1: Loading movie ratings dataset ================
%  You will start by loading the movie ratings dataset to understand the
%  structure of the data.
%  
fprintf('Loading movie ratings dataset.\n\n');

%  Load data
load ('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i




%  From the matrix, we can compute statistics like average rating.
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', ...
        mean(Y(1, R(1, :))));

%  We can "visualize" the ratings matrix by plotting it with imagesc
imagesc(Y);
ylabel('Movies');
xlabel('Users');

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ============ Part 2: Collaborative Filtering Cost Function ===========
%  You will now implement the cost function for collaborative filtering.
%  To help you debug your cost function, we have included set of weights
%  that we trained on that. Specifically, you should complete the code in 
%  cofiCostFunc.m to return J.

%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
load ('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 0);
           
fprintf(['Cost at loaded parameters: %f '...
         '\n(this value should be about 22.22)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ============== Part 3: Collaborative Filtering Gradient ==============
%  Once your cost function matches up with ours, you should now implement 
%  the collaborative filtering gradient function. Specifically, you should 
%  complete the code in cofiCostFunc.m to return the grad argument.
%  
fprintf('\nChecking Gradients (without regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction;

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ========= Part 4: Collaborative Filtering Cost Regularization ========
%  Now, you should implement regularization for the cost function for 
%  collaborative filtering. You can implement it by adding the cost of
%  regularization to the original cost computation.
%  

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 1.5);
           
fprintf(['Cost at loaded parameters (lambda = 1.5): %f '...
         '\n(this value should be about 31.34)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ======= Part 5: Collaborative Filtering Gradient Regularization ======
%  Once your cost matches up with ours, you should proceed to implement 
%  regularization for the gradient. 
%

%  
fprintf('\nChecking Gradients (with regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction(1.5);

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ============== Part 6: Entering ratings for a new user ===============
%  Before we will train the collaborative filtering model, we will first
%  add ratings that correspond to a new user that we just observed. This
%  part of the code will also allow you to put in your own ratings for the
%  movies in our dataset!
%
movieList = loadMovieList();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings(1) = 4;

% Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings(98) = 2;

% We have selected a few movies we liked / did not like and the ratings we
% gave are as follows:
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ================== Part 7: Learning Movie Ratings ====================
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users
%

fprintf('\nTraining collaborative filtering...\n');

%  Load data
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];


% Dividing rating dataset to Training, Cross-Validation and Test sets
% in 60%, 20% and 20%, respectively.
[nmovie, nuser]=size(Y);

f_random=random('uniform',0, 1, nuser, 1);
[u_tr, v]=find(f_random<0.6);
[u_cv, v]=find(f_random>=0.6 & f_random<0.8);
[u_te, v]=find(f_random>=0.8);

Ytr=Y(:, u_tr); % Training set
Ycv=Y(:, u_cv); % Cross Validation set
Vte=Y(:, u_te); % Test set

Rtr=R(:, u_tr); % Training set
Rcv=R(:, u_cv); % Cross Validation set
Rte=R(:, u_te); % Test set


%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Ytr, Rtr);

%  Useful Values
num_users = size(Ytr, 2);
num_movies = size(Ytr, 1);
num_features = 20;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Ytr, Rtr, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% ================== Part 8: Learning curve===========================

nu_tr = size(Ytr, 2);

nu_cv = size(Ycv, 2);

rand_tr=random('uniform',0, 1, nu_tr, 1);
rand_cv=random('uniform',0, 1, nu_cv, 1);

u_tr={}; % Indecies cell array for training data
u_cv={}; % Indecies cell array for cross-validation data

Ytrc={}; % Training data cell
Rtrc={}; % Indicator data cell

Ycvc={}; % cross-validation data cell
Rcvc={}; % Indicator data cell for cross validation data

Jtr=zeros(10,1); % Cost function 
Jcv=zeros(10,1); % Cost function 

Str=zeros(10,1); % Success rate
Scv=zeros(10,1); % Success rate

num_features = 20;

for i=1:10
    alpha=i*0.1;
    [u_tr{i}, v]=find(rand_tr<alpha);
    Ytrc{i}= Ytr(:,u_tr{i});
    Rtrc{i}= Rtr(:,u_tr{i});
    
    [u_cv{i}, v]=find(rand_cv<alpha);
    Ycvc{i}= Ycv(:,u_cv{i});
    Rcvc{i}= Rcv(:,u_cv{i});
    
    
    %  Normalize Ratings
    [Ynorm, Ymean] = normalizeRatings(Ytr, Rtr);

    %  Useful Values
    nu_tr = size(Ytrc{i}, 2);
    nm_tr = size(Ytrc{i}, 1);
    
    nu_cv = size(Ycvc{i}, 2);
    nm_cv = size(Ycvc{i}, 1);
    
    
    

    % Set Initial Parameters (Theta, X)
    Xtr = randn(nm_tr, num_features);
    Thetatr = randn(nu_tr, num_features);
    
    Thetacv = randn(nu_cv, num_features);
    

    initial_parameters_tr = [Xtr(:); Thetatr(:)];
    initial_parameters_cv = [Thetacv(:)];

    % Set options for fmincg
    options = optimset('GradObj', 'on', 'MaxIter', 100);

    % Set Regularization
    lambda = 10;
    theta_tr = fmincg (@(t)(cofiCostFunc(t, Ytrc{i}, Rtrc{i}, nu_tr, nm_tr, ...
                                    num_features, lambda)), ...
                    initial_parameters_tr, options);
                
    Jtr(i)=cofiCostFunc(theta_tr, Ytrc{i}, Rtrc{i}, nu_tr, nm_tr, ...
                                    num_features, 0);
                                
                                
    Str(i)=sucess(theta_tr, Ytrc{i}, Rtrc{i}, nu_tr, nm_tr,num_features);
                                
                                
    theta_cv = fmincg (@(t)(cofiCostFuncCV(t, Xtr, Ycvc{i}, Rcvc{i}, nu_cv, nm_cv, ...
                                    num_features, lambda)), ...
                    initial_parameters_cv, options);
                
    Jcv(i)=cofiCostFuncCV(theta_cv, Xtr,  Ycvc{i}, Rcvc{i}, nu_cv, nm_cv, ...
                                    num_features, 0);      
    theta_cv=[theta_cv ;Xtr(:)];
    Scv(i)=sucess(theta_cv, Ycvc{i}, Rcvc{i}, nu_cv, nm_cv,num_features);
    
end

subplot(121)
plot(0.1:0.1:1, [Jtr, Jcv]);
xlabel('Precent of traing data')
ylabel('Error')
legend('Jtr', 'Jcv')

subplot(122)
plot(0.1:0.1:1, [Str, Scv]);
xlabel('Precent of data')
ylabel('Success rate')
legend('Str', 'Scv')



%% ================== Part 8: Evaluation ===============================
%  After training the model, we evauate it with cross-validation set

num_users = size(Ytr, 2);
num_movies = size(Ytr, 1);
num_features = 10;

[Jtr, v]=cofiCostFunc(theta, Ytr, Rtr, num_users, num_movies, ...
        num_features, 0);
    
fprintf('Cost for the Training set is %d \n', Jtr);


[Jcv, v]=cofiCostFunc(theta, Ycv, Rcv, num_users, num_movies, ...
        num_features, 0);
    
fprintf('Cost for the Cross-Validations Set is %d \n', Jcv);
    



%% ================== Part 8: Recommendation for you ====================
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%

p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end
