% file: elem4_9.m
% function to form 9-node element stiffness matrix

function [ke] = elem4_9(node,x,y,gauss,weight,young,pr,e)

ke = zeros(18,18);
% compute element stiffness
    for j=1:9
       je=node(j,e);xe(j)=x(je);ye(j)=y(je);
    end
    for i=1:length(gauss)
     for j=1:length(gauss)
      psi=gauss(i);
      eta=gauss(j);
      [NJpsi,NJeta]=deriv4_9(psi,eta); 
      xpsi=NJpsi*xe'; ypsi=NJpsi*ye'; xeta=NJeta*xe'; yeta=NJeta*ye';
      Jinv=[yeta,-ypsi;-xeta,xpsi];
      jcob=xpsi*yeta-xeta*ypsi;
      NJdpsieta=[NJpsi;NJeta];
      NJdxy=Jinv*NJdpsieta;
      BJ=zeros(3,18);
      BJ(1,1:2:17)=NJdxy(1,1:9)  ;BJ(2,2:2:18)=NJdxy(2,1:9);
      BJ(3,1:2:17)=NJdxy(2,1:9)  ;BJ(3,2:2:18)=NJdxy(1,1:9);
      
  %plane stress
  %      fac=young(e)/(1.-pr(e)^2);  
  %      C=fac*[1.,      pr(e),   0;
  %             pr(e),   1.,      0;
  %             0,       0,       (1.-pr(e))/2];
        


  % plane strain
        fac=young(e)/((1.+pr(e))*(1.-2.*pr(e)));  
        C=fac*[1.-pr(e),  pr(e),   0;
               pr(e),   1.-pr(e),  0;
               0,           0, (1.-2*pr(e))/2];
 
 

      ke=ke+BJ'*C*BJ/jcob*weight(i)*weight(j); 
    end   % end j
   end  % end i






