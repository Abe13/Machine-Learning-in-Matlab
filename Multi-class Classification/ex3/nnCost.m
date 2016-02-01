function J=nnCost(y, h, num_labels)


J=0;
m=size(y, 1);

for k=1:num_labels
    
    yk=(y==k);
    % yk size = (m, 1)
    
    Jk=-1/m * (yk'*log(h(:,k))+(1-yk')*log(h(:,k)));
    J=J+Jk;
end

