function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));                 
                 
% Setup some useful variables
m = size(X, 1);

YY = zeros(m, num_labels);

for i = 1:m,
  YY(i, y(i)) = 1;
endfor

% disp(YY);  
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1

A1 = [ones(m, 1) X]; % m x (1 + input_layer_size)

% m x (1 + input_layer_size) * (hidden_layer_size x (input_layer_size + 1))'
Z2 = A1 * Theta1'; % m x hidden_layer_size
A2 = sigmoid(Z2); % m x hidden_layer_size
A2 = [ones(size(A2, 1), 1) A2]; % m x (hidden_layer_size + 1)

% m x (hidden_layer_size + 1) * (num_labels x (hidden_layer_size + 1))'
Z3 = A2 * Theta2'; % m x num_labels
A3 = sigmoid(Z3); % m x num_labels

J = (1 / m) * sum(sum(- YY .* log(A3) - (1 - YY) .* log(1 - A3)));

reg = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end) .* Theta1(:,2:end))) + sum(sum(Theta2(:,2:end) .^ 2)));

J = J + reg;

% Part 2

Del1 = zeros(size(Theta1));
Del2 = zeros(size(Theta2));

for i = 1:m,
  A1 = [1 ; X(i, :)']; % 401 x 1
  
  % Theta1 25 x 401
  
  Z2 = Theta1 * A1; % 25 x 1
  A2 = [1 ; sigmoid(Z2)]; % 26 x 1
  
  % Theta2 10 x 26
  
  Z3 = Theta2 * A2; % 10 x 1
  A3 = sigmoid(Z3); % 10 x 1
  
  % cost 
  %{
  for k = 1:num_labels,  
    J = J + sum(- (YY(i, k) * log(A3(k, 1))) - ((1 - YY(i, k)) * log(1 - A3(k, 1))));
  endfor
  %}
  
  
  yy = ([1:num_labels]==y(i))';
  del3 = A3 - yy; % 10 x 1 
  % del3 = A3 - YY(i)'; % 10 x 1 
  
  % 25 x 1 = 26 x 10 * 10 x 1 .* 25 x 1
  del2 = (Theta2' * del3) .* [1; sigmoidGradient(Z2)];
  del2 = del2(2:end);
  % del2 = (Theta2' * del3) .* sigmoidGradient(A_2);
  
  % 25 x 401 == 25 x 1 * 1 x 401
  Del1 = Del1 + del2 * A1';  
  % Del1 = Del1 + del2(2:end, :) * A_1';
  
  
  % 10 x 26 == 10 x 1 * 1 x 26
  Del2 = Del2 + del3 * A2';  
endfor

Theta1_grad = Del1 / m + (lambda / m ) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)] ;
Theta2_grad = Del2 / m + (lambda / m ) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)] ;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
