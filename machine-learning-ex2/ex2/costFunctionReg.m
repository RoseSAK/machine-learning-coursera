function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
%lambdas = ones(length(y));
%lambdas(1) = m; % so m/m cancels 

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); % size = 28 rows, 1 column
% X size = 118 rows, 28 columns 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta); % size h = 118 rows, 1 column 

J_unreg = 1/m * ((log(h')*-y) - (log(1-h')*(1-y)));
theta(1) = 0; 
J_reg = (lambda/(2*m))*(theta'*theta);
J = J_unreg + J_reg; 

grad_unreg = ((h - y)'*X)/m;
grad_reg = theta*(lambda/m);
grad = grad_unreg' + grad_reg;

% =============================================================

end
