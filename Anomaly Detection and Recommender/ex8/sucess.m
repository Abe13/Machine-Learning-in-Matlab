function s=success(params, Y, R, nu, nm, nf)
% the function s=sucess(theta_tr, Ytrc{i}, Rtrc{i}, nu_tr, nm_tr) returns
% the success rate


X = reshape(params(1:nm*nf), nm, nf);
Theta = reshape(params(nm*nf+1:end), nu, nf);

H=X*Theta';


Yest=5*(H>=4.5) + 4*(H<4.5 & H>=3.5) + 3*(H<3.5 & H>=2.5)+...
    2*(H<2.5 & H>=1.5)+1*(H<=1.5);
    
s=sum(sum(R.*(Y==Yest)))/sum(sum(R));
