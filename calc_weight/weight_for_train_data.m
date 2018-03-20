function ret = weight_for_train_data();

  % path for mosek
  addpath /home/mosek/8/toolbox/r2014a

for data = 0:4
    for fold = 0:4
        dirname = sprintf('../data/data_%d/fold_%d/',data,fold);
        fprintf('\t[*] Starting %s\n',dirname);

        disp('\t[*] Loading file...');
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

        disp('\t[*] Clac LaplacianMatrix...');
        Lap = LaplacianMatrix(Wlink);
        PLP = Prd*Lap*Prd';
        Q = PLP + PLP';

        % solve quadprog with mosek
        lambda = [100. 10. 1. .1 .01 .001 .0001];
        % lambda = [10000. 1000. 100. 10. 1. .1];
        % lambda = [20. 19. 18. 17. 16. 15. 14. 13. 12. 11. 10. 9. 8. 7. 6. 5. 4. 3. 2. 1.];

        [not_use len] = size(lambda);
        disp('\t\t[-] Done Setting...');
        disp('\t[*] Solving quadprog with mosek...')
        for j = 1:len
            disp(lambda(j));
            compute_weight(lambda(j), Y, Prd, Q, dirname);
        end

        disp('\t\t[-] Solved');

        fprintf('\t[*] Done fold_%d\n',fold);
    end
    fprintf('[*] Done data_%d\n',data);
end
% end
