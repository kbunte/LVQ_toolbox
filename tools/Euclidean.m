function D = Euclidean(A, B)
% computes the Euclidean distance
%
%   D = squaredEuclidean(X) returns the Euclidean distance matrix of data in rows of X 
%   D = squaredEuclidean(X, Y) returns the distance matrix with all distances between the points in X and Y.
%
if nargin == 1 % means that one matrix
    D = bsxfun(@plus, sumsquared(A,2), bsxfun(@minus, sumsquared(A,2).', 2*A*A.')); % 2*(Y*Y.')
else    
    D = bsxfun(@plus, sumsquared(A,2), bsxfun(@minus, sumsquared(B,2).', 2*A*B.'));
end
D = sqrt(max(D,0));