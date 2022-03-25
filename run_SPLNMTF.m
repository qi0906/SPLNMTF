%-------------------------------------------------------------------------
% Function for exporting predictions 
%-------------------------------------------------------------------------

function run_SPLNMTF(k, R, A, label_list, max_iter, initialization)

ext = [num2str(k(1)) '_' num2str(k(2)) '_' num2str(k(3))];

[data_R,data_H,data_G] = concat_data(R,A,k,initialization);
[~,H, G] = SPLNMTF(data_R,data_H,data_G,max_iter);
file_list = {['./results/patients_final_' ext],...
             ['./results/genes_final_' ext],...
             ['./results/drugs_final_' ext]};

% Exporting clusters 
compute_clusters_ssnmtf(G,label_list, file_list);

% Writing connectivity matrix into file 
dlmwrite(['./results/gclust-gclust_' ext '.mtrx'], full(H{2,2}), 'delimiter', ',');

% Exporting predictions 
export_significant_associations(G, H, label_list, [1,2], ['./results/patient-gene_pred_' ext '.txt'], 'mix')
export_significant_associations(G, H, label_list, [2,2], ['./results/gene-gene_pred_' ext '.txt'], 'mix')
export_significant_associations(G, H, label_list, [2,3], ['./results/gene-drug_pred_' ext '.txt'], 'mix')
