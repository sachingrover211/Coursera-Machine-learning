function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
z = X*theta;
h = sigmoid(z);
s = size(theta, 1);

%for j = 1:m
%	grad(1) = grad(1) + (X(j, 1)*(h(1)- y(1)));
%end

%for i = 2:s
%	for j = 1:m
%		grad(i) = grad(i) + (X(j, i)'*(h(i) - y(i)));
%	end
%	grad(i) = grad(i) + lambda*theta(i);
%end

%grad = grad/m;

%there is something wrong with the code above. Could not figure it out due to lack of test cases, even in discussion forum.
%took some help for the two line code below for gradient from - https://github.com/cnauroth/machine-learning-class
thetaWithZero = [0; theta(2:s)];
grad = (X'*(h-y) + lambda*thetaWithZero)/m;

J = -(y'*log(h) + (1-y)'*log(1 - h))/m + lambda*(theta(2:s)'*theta(2:s))/(2*m);

% =============================================================

end
