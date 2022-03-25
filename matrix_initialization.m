%--------------------------------------------------------------------------------------------------------------
% Function for initializing G matrices by using different initialaztion strategies: {random,random_acol}
%--------------------------------------------------------------------------------------------------------------
% [Input]: 
%   R: <2D Cell array>, r(node types) x r(node types) blocks 
%   indx: <integer>, index of a G matrix 
%   ni, ki: <integer>, size and rank of G_indx matrix 
%   strategy: <string>, {'random_acol','nnmf','random'}
% [Output]: 
%   G_indx: <matrix>, block matrix G{indx} 
%--------------------------------------------------------------------------
function G = matrix_initialization(R,indx,n_indx,k_indx,strategy)
 rand('seed',20)
if strcmp(strategy,'random')  
    G = sparse(rand(n_indx,k_indx));
    
elseif strcmp(strategy,'random_acol')
    num = length(R);
    G = sparse(1e-10*ones(n_indx,k_indx)); 
    for i=1:num
        if i ~= indx
            s = size(R{indx,i});   
            p = round(s(2)/3); 
            for k=1:k_indx
                perm = randperm(s(2)); 
                col = R{indx,i}(:,perm(1:p));  
                G(:,k) = G(:,k) + mean(col,2); 
            end
        end
    end
    G = G./(num-1);  
    
elseif strcmp(strategy,'nnmf')
    num = length(R);  
    G = sparse(1e-10*ones(n_indx,k_indx)); 
    for i=1:num
        if i ~= indx
            [W,~] = nnmf(R{indx,i},k_indx); 
            G = G + sparse(W);
        end
    end
    G = G./(num-1);
    
else
    error('--Wrong type of initialization strategy. Possible: {row,col,mix}');
end
