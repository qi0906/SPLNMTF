%-----------------------------------------------------------------------
% Function for 5-fold cross-validation
% ----------------------------------------------------------------------

function cross_validation(k, R, A, max_iter, initialization)

[data_R,data_H,data_G] = concat_data(R,A,k,initialization);

R12 = data_R{1,2}; 
R23 = data_R{2,3};
R12_mat = full(R12);
R23_mat = full(R23);

% % Save raw relational data
% csvwrite('./cross_val_data/raw_R23_mat.csv',R23_mat)

%%%%% Randomly assign 0 to R12_mat %%%%%
% Find all 1's in R12
temp_mat1 = [];
temp_mat2 = [];
for m1 = 1:size(R12_mat,1)
    for n1 = 1:size(R12_mat,2)
        if R12_mat(m1,n1) == 1
            temp_mat1 = [temp_mat1,m1];
            temp_mat2 = [temp_mat2,n1];
        end
    end
end
R12_temp_mat = [temp_mat1;temp_mat2];

%%% Shuffle the index randomly %%%
rand('seed',20)
randIndex1 = randperm(size(R12_temp_mat,2));
R12_temp_mat_new = R12_temp_mat(:,randIndex1);


%%%%% Randomly assign 0 to R23_mat %%%%%
% Find all 1's in R23
temp_mat3 = [];
temp_mat4 = [];
for m2 = 1:size(R23_mat,1)
    for n2 = 1:size(R23_mat,2)
        if R23_mat(m2,n2) == 1
            temp_mat3 = [temp_mat3,m2];
            temp_mat4 = [temp_mat4,n2];
        end
    end
end
R23_temp_mat = [temp_mat3;temp_mat4];

%%% Shuffle the index randomly %%%
rand('seed',20)
randIndex2 = randperm(size(R23_temp_mat,2));
R23_temp_mat_new = R23_temp_mat(:,randIndex2);


k_folds = 5;
R12_fold = round(size(R12_temp_mat_new,2) / k_folds);
R23_fold = round(size(R23_temp_mat_new,2) / k_folds);

for k = 1:5
    fprintf('%d fold CV \n\n',k)
    new_R12_mat = R12_mat;
    new_R23_mat = R23_mat;
    new_f1 = [];
    new_f2 = [];
    new_g1 = [];
    new_g2 = [];
    if k ~= k_folds
        for f = R12_temp_mat_new(:,((k-1) * R12_fold) + 1 : k * R12_fold)
        new_R12_mat(f(1),f(2)) = 0;
        new_f1 = [new_f1,f(1)];
        new_f2 = [new_f2,f(2)];
        end
        fin_f = [new_f1;new_f2];
        
        
        for g = R23_temp_mat_new(:,((k-1) * R23_fold) + 1 : k * R23_fold)
        new_R23_mat(g(1),g(2)) = 0;
        new_g1 = [new_g1,g(1)];
        new_g2 = [new_g2,g(2)];
        end
        fin_g = [new_g1;new_g2];
        
    else
        for f = R12_temp_mat_new(:,((k-1) * R12_fold) + 1 : end)
        new_R12_mat(f(1),f(2)) = 0;
        new_f1 = [new_f1,f(1)];
        new_f2 = [new_f2,f(2)];
        end
        fin_f = [new_f1;new_f2];
        

        for g = R23_temp_mat_new(:,((k-1) * R23_fold) + 1 : end)
        new_R23_mat(g(1),g(2)) = 0;
        new_g1 = [new_g1,g(1)];
        new_g2 = [new_g2,g(2)];
        end
        fin_g = [new_g1;new_g2];
        
    end
    
    data_R{1,2} = new_R12_mat;
    data_R{2,3} = new_R23_mat;
    
    [re_R, ~, ~] = SPLNMTF(data_R,data_H,data_G,max_iter);
    temp_R23_mat = re_R{2,3};
    
    if k==1
    disp('Save the results of 1-fold CV')
    csvwrite('./cross_val_data/R23_mat1.csv',temp_R23_mat)
    csvwrite('./cross_val_data/cv_R23_mat1.csv',new_R23_mat)
    csvwrite('./cross_val_data/pv_index1.csv',fin_g)
    end
    
    if k==2
    disp('Save the results of 2-fold CV')
    csvwrite('./cross_val_data/R23_mat2.csv',temp_R23_mat)
    csvwrite('./cross_val_data/cv_R23_mat2.csv',new_R23_mat)
    csvwrite('./cross_val_data/pv_index2.csv',fin_g)
    end
    
    if k==3
    disp('Save the results of 3-fold CV')
    csvwrite('./cross_val_data/R23_mat3.csv',temp_R23_mat)
    csvwrite('./cross_val_data/cv_R23_mat3.csv',new_R23_mat)
    csvwrite('./cross_val_data/pv_index3.csv',fin_g)
    end
    
    if k==4
    disp('Save the results of 4-fold CV')
    csvwrite('./cross_val_data/R23_mat4.csv',temp_R23_mat)
    csvwrite('./cross_val_data/cv_R23_mat4.csv',new_R23_mat)
    csvwrite('./cross_val_data/pv_index4.csv',fin_g)
    end
    
    if k==5
    disp('Save the results of 5-fold CV')
    csvwrite('./cross_val_data/R23_mat5.csv',temp_R23_mat)
    csvwrite('./cross_val_data/cv_R23_mat5.csv',new_R23_mat)
    csvwrite('./cross_val_data/pv_index5.csv',fin_g)
    end
end
