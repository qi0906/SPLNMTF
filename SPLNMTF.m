%-----------------------------------------------------------------
% Function for the main idea of SPLNMTF
% ----------------------------------------------------------------
function [R,H,G] = SPLNMTF(R,H,G,max_iter)

disp('Start')

G1 = G{1,1};  % n1 * K1
G2 = G{1,2};  % n2 * K2
G3 = G{1,3};  % n3 * K3
G1_mat = full(G1);
G2_mat = full(G2);
G3_mat = full(G3);

% clustering matrix
H12 = H{1,2};
H23 = H{2,3};
H12_mat = full(H12);
H23_mat = full(H23);

% R12 is the patient*gene, R23 is the gene*drug
R12 = R{1,2}; 
R23 = R{2,3}; 
R12_mat = full(R12);
R23_mat = full(R23);

% Laplace matrix L2
R22 = R{2,2}; 
R22_mat = full(R22);
temp_D2 = eye(size(R22_mat));
for m = 1:size(temp_D2)
    temp_D2(m,m) = sum(R22_mat(m,:)); 
end
D2 = temp_D2;
A2 = R22_mat;
L2 = D2 - A2;

% Laplace matrix L3
R33 = R{3,3};
R33_mat = full(R33);
temp_D3 = eye(size(R33_mat));
for n = 1:size(temp_D3)
    temp_D3(n,n) = sum(R33_mat(n,:));    
end
D3 = temp_D3;
A3 = R33_mat;
L3 = D3 - A3;


% Initialize SPL weight matrices W12 and W23
W12 = ones(size(R12_mat));
W23 = ones(size(R23_mat));
gamma = 1;

for lambda =[0.9,0.8,0.6,0.5,0.3,0.1,1e-40,1e-40;0.9,0.8,0.6,0.5,0.3,0.1,1e-40,1e-40]
    fprintf('lambda is %d,%d\n\n',lambda(1),lambda(2)) 
    for iter=1:max_iter
        
        % updata G1
        new_R12 = W12 .* R12_mat;
        new_G12 = new_R12 * G2_mat * H12_mat';
        for i = 1:size(G1_mat,1)
            temp1 = (new_G12 ./ (G1_mat * H12_mat * (G2_mat)' * sparse(diag(W12(i,:))) * G2_mat * H12_mat' + 1e-8)).^0.5;
            G1_mat(i,:) = G1_mat(i,:) .* temp1(i,:);
        end
        
        % updata G2
        new_R23 = W23 .* R23_mat;
        new_W12 = W12';
        L2_sub = (abs(L2) - L2)/2;
        L2_add = (abs(L2) + L2)/2;
        Numerator1 = new_R12' * G1_mat * H12_mat + new_R23 * G3_mat * H23_mat' + L2_sub * G2_mat;
        for j = 1:size(G2_mat,1)
            temp2 = (Numerator1 ./ (G2_mat * H12_mat' * (G1_mat)' * sparse(diag(new_W12(j,:))) * G1_mat * H12_mat + G2_mat * H23_mat * G3_mat' * sparse(diag(W23(j,:))) * G3_mat * H23_mat' + L2_add * G2_mat + 1e-8)).^0.5;
            G2_mat(j,:) = G2_mat(j,:) .* temp2(j,:);     
        end
        
        % updata G3
        new_W23 = W23';
        L3_sub = (abs(L3) - L3)/2;
        L3_add = (abs(L3) + L3)/2;
        Numerator2 = new_R23' * G2_mat * H23_mat + L3_sub * G3_mat; 
        for t = 1:size(G3_mat,1)
            temp3 = (Numerator2 ./ (G3_mat * H23_mat' * (G2_mat)' * sparse(diag(new_W23(t,:))) * G2_mat * H23_mat + L3_add * G3_mat + 1e-8)).^0.5;
            G3_mat(t,:) = G3_mat(t,:) .* temp3(t,:);
        end
        
        % updata H12
        Numerator3 = G1_mat' * new_R12 * G2_mat;
        for h1 = 1:size(H12_mat,1)
            temp4 = (Numerator3 ./ (G1_mat'* G1_mat * H12_mat * G2_mat' * sparse(diag(W12(h1,:))) * G2_mat + 1e-8)).^0.5;
            H12_mat(h1,:) = H12_mat(h1,:) .* temp4(h1,:);
        end
        
        % updata H23
        Numerator4 = G2_mat' * new_R23 * G3_mat;
        for h2 = 1:size(H23_mat,1)
            temp5 = (Numerator4 ./ (G2_mat' * G2_mat * H23_mat * G3_mat' * sparse(diag(W23(h2,:))) * G3_mat + 1e-8)).^0.5;
            H23_mat(h2,:) = H23_mat(h2,:) .* temp5(h2,:);
        end
        
        % stop condition
        if mod(iter+1,2) == 0
            J_old = (norm(W12 .* (R12_mat - G1_mat * H12_mat * G2_mat'),'fro'))^2 + (norm(W23 .* (R23_mat - G2_mat * H23_mat * G3_mat'),'fro'))^2;
        end
        if mod(iter,2) == 0  
            J_new = (norm(W12 .* (R12_mat - G1_mat * H12_mat * G2_mat'),'fro'))^2 + (norm(W23 .* (R23_mat - G2_mat * H23_mat * G3_mat'),'fro'))^2;
            obs = abs(J_new - J_old)/abs(J_old);
            
            if obs < 10e-5
            break
            end
        end
    end
    
    % R12 and R23 reconstructed
    out_R12 = G1_mat * H12_mat * G2_mat';
    out_R23 = G2_mat * H23_mat * G3_mat';
    
    % Self-paced learning of R12
    for I = 1:size(R12_mat,1)
        for J = 1:size(R12_mat,2)
            
            if power(out_R12(I,J) - R12_mat(I,J),2) <= 1/((lambda(1) + 1/gamma)^2)
                W12(I,J) = 1;

            elseif power(out_R12(I,J) - R12_mat(I,J),2) >= 1/(lambda(1)^2)
                W12(I,J) = 0;

            else
                W12(I,J) = gamma * (1/sqrt(power(out_R12(I,J) - R12_mat(I,J),2))-lambda(1));

            end
        end
    end

    % Self-paced learning of R23 
    for M = 1:size(R23_mat,1)
        for N = 1:size(R23_mat,2)
            
            if power(out_R23(M,N) - R23_mat(M,N),2) <= 1/((lambda(2) + 1/gamma)^2)
                W23(M,N) = 1;
             
 
            elseif power(out_R23(M,N) - R23_mat(M,N),2) >= 1/(lambda(2)^2)
                W23(M,N) = 0;

            else
                W23(M,N) = gamma * (1/sqrt(power(out_R23(M,N) - R23_mat(M,N),2))-lambda(2));

            end
        end
    end
end

% Converiting back to block matrices (cells)  
R{1,2} = out_R12;
R{2,3} = out_R23;

H{1,2} = H12_mat;
H{2,3} = H23_mat;
G{1,1} = G1_mat;
G{1,2} = G2_mat;
G{1,3} = G3_mat;
