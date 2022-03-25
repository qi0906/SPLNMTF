diary  mydiary
diary on;
tic

fprintf('\nSTART TIME:    %s\n\n', datestr(now));

k = [6, 35, 45];
max_iter = 1000;
initialization = 'random_acol';

fprintf('\n\n\n')

ext = [num2str(k(1)) '_' num2str(k(2)) '_' num2str(k(3))];

adj_list = {'./data/OVARIAN_PATIENT-PATIENT_mtrx.txt','./data/OVARIAN_GENE-GENE_mtrx.txt','./data/OVARIAN_DRUG-DRUG_mtrx.txt'};
rel_file = './data/OVARIAN_RELATIONS_mtrx.txt';

% create block matrices 
[R, A, label_list] = block_matrices(adj_list, rel_file);

% % 5-fold cross-validation
cross_validation(k, R, A, max_iter, initialization)

% %Calculate evaluation metrics, such as AUC, etc.
disp('Start calculating evaluation metrics！！！！！！！！！！！！！！！！')
unix(['python Metric.py ./cross_val_data/R23_mat1.csv ./cross_val_data/cv_R23_mat1.csv ./cross_val_data/pv_index1.csv ./cross_val_data/R23_mat2.csv ./cross_val_data/cv_R23_mat2.csv ./cross_val_data/pv_index2.csv ./cross_val_data/R23_mat3.csv ./cross_val_data/cv_R23_mat3.csv ./cross_val_data/pv_index3.csv ./cross_val_data/R23_mat4.csv ./cross_val_data/cv_R23_mat4.csv ./cross_val_data/pv_index4.csv ./cross_val_data/R23_mat5.csv ./cross_val_data/cv_R23_mat5.csv ./cross_val_data/pv_index5.csv ./cross_val_data/raw_R23_mat.csv']);


% % % Special Notes: 1. When performing a 5-fold cross-validation experiment, you can comment out the following functions. i.e. "run_SPLNMTF".% % %
% % % 2. When conducting drug repositioning experiments, you can comment out the above two functions. Namely "5-fold cross_validation" and "calculating evolution metrics".% % %


% % % run SPLNMTF and  exporting predictions 
% run_SPLNMTF(k, R, A, label_list, max_iter, initialization);

toc
diary off;

