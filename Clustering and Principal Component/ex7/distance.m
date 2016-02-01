function dis=distance(a,b)

m=size(a,1);

d=a-ones(m,1)*b;
dis=(sum(d'.^2)).^0.5;
