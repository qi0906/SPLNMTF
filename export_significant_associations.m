%--------------------------------------------------------------------------
% Function for reconstructing relation matrices -> only significant values
% -------------------------------------------------------------------------

function export_significant_associations(G, H, label_list, pair_ind, sim_file, exp_type)

tol = 1e-5; % scores below tol are considered zero 
fprintf('\n\n');
fprintf('---Creating reconstructed relation matrix:\n');

gene_labels = label_list{2};

Rapprox = G{pair_ind(1)}*H{pair_ind(1),pair_ind(2)}*G{pair_ind(2)}';
Rapprox = Rapprox.*(Rapprox > tol);
Rapprox = centric_rule(Rapprox,exp_type);
if ( pair_ind(1) == pair_ind(2) )
    Rapprox = triu(Rapprox);  
end;
fprintf('---Finished');


% New export way (combination of row- and  column-centric rules)
[indx_i,indx_j] = find(Rapprox);
indx_val = find(Rapprox);
fprintf('\n\n\n');
fprintf('--Exporting associations:');

fid = fopen(sim_file,'w');  
fprintf(fid, 'Gene1 | Drug2 | Score\n');
for i=1:length(indx_val)
    fprintf(fid,'%s %s %f\n',label_list{pair_ind(1)}{indx_i(i)},label_list{pair_ind(2)}{indx_j(i)},full(Rapprox(indx_val(i))));
    if (mod(i,500)==0)  
    end;
end;

fprintf('\n\n\n');
fprintf('--Writing significant associatons finished!\n\n');
