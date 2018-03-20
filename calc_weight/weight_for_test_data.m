function ret = weight_for_test_data(lambda);

% path for mosek
addpath /home/mosek/8/toolbox/r2014a


% test data
for data = 0:4
    dirname = sprintf('../data/data_%d/',data);
    fprintf('[*] Starting test data',dirname);

    disp('[*] Loading file...');
    y = strcat(dirname,'truelabel.csv');
    Y = csvread(y); % get the true labels of the labeled data

    prd = strcat(dirname,'predictions.csv');
    Prd = csvread(prd)
    % Prd1 = csvread(prd);
    % prd = strcat(dirname,'predictions_nn.csv');
    % Prd2 = csvread(prd);
    % Prd = vertcat(Prd1,Prd2); % get the predictions of the base classifiers
    w = strcat(dirname,'w_link/w_link_0.csv');
    w1 = csvread(w);
    w = strcat(dirname,'w_link/w_link_1.csv')
    w2 = csvread(w);
    w = strcat(dirname,'w_link/w_link_2.csv');
    w3 = csvread(w);
    w = strcat(dirname,'w_link/w_link_3.csv')
    w4 = csvread(w);
    Wlink = vertcat(w1,w2,w3,w4); % get the kernel matrix link matrix

    disp('[*] Clac LaplacianMatrix...');
    Lap = LaplacianMatrix(Wlink);
    PLP = Prd*Lap*Prd';
    Q = PLP + PLP';

    % solve quadprog with mosek
    % lambda = 11.0;

    disp('\t[-] Done Setting...');
    disp('[*] Solving quadprog with mosek...')

    disp(lambda(data+1));
    compute_weight(lambda(data+1), Y, Prd, Q, dirname);

    disp('\t[-] Solved');

    disp('[*] Done test data\n');
end
% end
