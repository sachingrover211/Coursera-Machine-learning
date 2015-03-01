function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
[l w] = size(z);
temp = reshape(z, l*w, 1);
temp = 1./(1 + exp(-temp));

g = reshape(temp, l, w);
% =============================================================

end
