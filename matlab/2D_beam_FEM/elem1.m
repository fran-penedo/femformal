% file: elem1.m


function [ke] = elem1(node,x,y,gauss,weight,young,pr,e)

% 2D Quad Element Stiffness Subroutine
ke = zeros(8,8);
one = ones(1,4);
psiJ=[-1,+1,+1,-1];etaJ=[-1,-1,+1,+1];
% compute element stiffness
    for j=1:4
       je=node(j,e);xe(j)=x(je);ye(j)=y(je);
    end
    for i=1:length(gauss)
     for j=1:length(gauss)
      psi=gauss(i);
      eta=gauss(j);  
      NJpsi=.25*psiJ.*(one+eta*etaJ);
      NJeta=.25*etaJ.*(one+psi*psiJ);
      xpsi=NJpsi*xe'; ypsi=NJpsi*ye'; xeta=NJeta*xe'; yeta=NJeta*ye';
      Jinv=[yeta,-ypsi;-xeta,xpsi];
      jcob=xpsi*yeta-xeta*ypsi;
      NJdpsieta=[NJpsi;NJeta];
      NJdxy=Jinv*NJdpsieta;
      BJ=zeros(3,8);
      BJ(1,1:2:7)=NJdxy(1,1:4)  ;BJ(2,2:2:8)=NJdxy(2,1:4);
      BJ(3,1:2:7)=NJdxy(2,1:4)  ;BJ(3,2:2:8)=NJdxy(1,1:4);
  
% plane stress
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