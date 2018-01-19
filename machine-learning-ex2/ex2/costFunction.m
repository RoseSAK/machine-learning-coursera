function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%h = X*theta;
%sqrErr = (h - y).^2;
%J = 1/(2*m)*sum(sqrErr) 

%h = sigmoid(X*theta);

%J = 1/m * ((log(h')*-y) - (log(1-h')*(1-y)));

%grad = ((h - y)'*X)/m 

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
