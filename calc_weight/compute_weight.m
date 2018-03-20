function weight = compute_weight(lambda, Y, Prd, Q, dirname);
% compute the weights for the base classifiers
%
% Input:
%       Cfr: the ensemble classifier, which contains multiple base
%       classifiers
% Output:
%       weight: the weights for the base classifiers in the ensemble


% M : set the size of classifiers
% N : set the size of training data
[M, N] = size(Prd)
NumVar = M + N;

% ---- Prepare QP ----
%equal constaint

Aeq = [ones(1,M),zeros(1,NumVar - M)];
beq = 1;
% lower and upper bounds
lb = zeros(NumVar,1);
% Ax <= b
AP = Prd';
for id = 1:N
    AP(id, :) = AP(id, :) * Y(id);
end
AP = -1 .* AP;
A = [ AP, -1 * eye(N) ];
b = -1 * ones(N ,1);

% H
H = zeros(NumVar,NumVar);
H(1:M, 1:M) = Q;
% f
% lambda = 1.0;
f = lambda * [zeros(M,1);ones(N, 1)];

% Optimization Using QP - MOSEK
options = optimset('Display','off','TolFun', 1e-100);
x0 = quadprog(H,f,A,b,Aeq,beq,lb,[],[],options);
weight = x0(1:M);
weight((weight <= 1e-6)) = 0;
filename = sprintf('%sweight/weight_lambda_%d%s',dirname,lambda,'.csv')
disp(filename)
csvwrite(filename,weight);


% end of function
