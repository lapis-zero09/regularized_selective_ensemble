function Lap = LaplacianMatrix(W)
% compute the laplacian matrix
%
% Input:
%       W: the line matrix
% Output:
%       Lap: the laplacian matrix


D = diag(sum(W,2));
N = D^(-0.5);
Lap = N *(D-W)* N;
