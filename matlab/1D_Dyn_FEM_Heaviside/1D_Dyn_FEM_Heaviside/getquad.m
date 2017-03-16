% Calculate quadrature points for cells based on xn

function [xq,wq] = getquad(x1,x2,nquad);


h  = x2 - x1;
[xcell,wcell] = quadrature(nquad);
wcell = wcell';
xq = x1 + (xcell+1)*h/2;
wq = wcell * h/2;
  
