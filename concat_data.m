%---------------------------------------------------------------
% Function for compliting matrices;
% --------------------------------------------------------------

function [R,H,G] = concat_data(R,A,k,initialization)

r = length(A); 

% Compliting relation matrix 
n = [];
for ii=1:r    
    R{ii,ii} = A{ii};
    n(ii) = length(A{ii});
end

% For of each data source 
fprintf('-Initializations of G matrices....\n');
for ii=1:r
    G{ii} = matrix_initialization(R,ii,n(ii),k(ii),initialization); 
    fprintf('Initialization of G[%d] matrix finished!\n',ii);
end
fprintf('-Initialization of G matrices finished!\n\n');


fprintf('-Initializations of H matrices....\n');
for ii=1:r
    for jj=1:r
        if (nnz(R{ii,jj}) ~= 0)
            H{ii,jj} = sparse(rand(k(ii),k(jj)));
            fprintf('Initialization of H[%d,%d] matrix finished!\n',ii,jj);
        else
            H{ii,jj} = sparse(zeros(k(ii),k(jj)));
        end
    end
end
fprintf('-Initialization of H matrices finished!\n\n');