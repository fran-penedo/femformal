function[erre]=dispnrm(node,nnode,x,y,e,disp,P,ebar,nubar,I,D,L)

if (nnode==4)
%gauss2=[.906179845938664,-.906179845938644,0,.538469310105683,-.538469310105683];
gauss2=[0.960289856498,-0.796666477414,-0.525532409916,-0.183434642496,0.183434642496,0.525532409916,0.796666477414,0.960289856498]; 	
%weight=[.236926885056189,.236926885056189,.568888888888889,.478628670499366,.478628670499366];
weight = [0.10122853629,0.222381034453,0.313706645878,0.362683783378,0.362683783378,0.313706645878,0.222381034453,0.10122853629];
one = ones(1,4);
psiJ=[-1,+1,+1,-1];etaJ=[-1,-1,+1,+1];
for j=1:4
    je=node(j,e);xe(j)=x(je);ye(j)=y(je);
    dispx(j)=disp(2*je-1);
    dispy(j)=disp(2*je);
end
erre=0;

for i=1:length(gauss2)
   for j=1:length(gauss2)
      eta=gauss2(i); psi=gauss2(j); 
      Npsi=.25*psiJ.*(one+eta*etaJ);
      Neta=.25*etaJ.*(one+psi*psiJ);
      xpsi=Npsi*xe'; ypsi=Npsi*ye'; xeta=Neta*xe'; yeta=Neta*ye';
      jcob=xpsi*yeta-xeta*ypsi;
      N=0.25*(one+psi*psiJ).*(one+eta*etaJ);
      xk=N*xe';
      yk=N*ye';
% exact displacements
      dispexactx=-P*yk/6/ebar/I*((6*L-3*xk)*xk+(2+nubar)*(yk*yk-D*D/4));
      dispexacty=P/6/ebar/I*(3*nubar*yk*yk*(L-xk)+D*D/4*(4+5*nubar)*xk+3*(L-xk/3)*xk*xk);

      %errorx=(N*dispx')-dispexactx;
      %errory=(N*dispy')-dispexacty;
      errorx=dispexactx-(N*dispx');
      errory=dispexacty-(N*dispy');
      ergaus=((errorx^2)+(errory^2))*weight(i)*weight(j)*jcob;
      erre=erre+ergaus;
   end   
end 

else
erre=0; 
gauss=[.906179845938664,-.906179845938644,0,.538469310105683,-.538469310105683];
weight=[.236926885056189,.236926885056189,.568888888888889,.478628670499366,.478628670499366];
% compute element stiffness
    for j=1:9
       je=node(j,e);xe(j)=x(je);ye(j)=y(je);
       dispx(j)=disp(2*je-1);
       dispy(j)=disp(2*je);
    end
    for i=1:5
     for j=1:5
      psi=gauss(i);
      eta=gauss(j);  
      NJpsi1=.25*(2*psi-1)*(eta^2-eta);
      NJeta1=.25*(psi^2-psi)*(2*eta-1);
      NJpsi2=.25*(2*psi+1)*(eta^2-eta);
      NJeta2=.25*(psi^2+psi)*(2*eta-1);
      NJpsi3=.25*(2*psi+1)*(eta^2+eta);
      NJeta3=.25*(psi^2+psi)*(2*eta+1);
      NJpsi4=.25*(2*psi-1)*(eta^2+eta);
      NJeta4=.25*(psi^2-psi)*(2*eta+1);
% Derivatives of middle nodes (nodes 5-8)
      NJpsi5=.5*(-2*psi)*(eta^2-eta); 
      NJeta5=.5*(1-psi^2)*(2*eta-1);
      NJpsi6=.5*(2*psi+1)*(1-eta^2);
      NJeta6=.5*(psi^2+psi)*(-2*eta);
      NJpsi7=.5*(-2*psi)*(eta^2+eta);
      NJeta7=.5*(1-psi^2)*(2*eta+1);
      NJpsi8=.5*(2*psi-1)*(1-eta^2);
      NJeta8=.5*(psi^2-psi)*(-2*eta);
% Derivatives of central node (node 9)
      NJpsi9=(-2*psi)*(1-eta^2);
      NJeta9=(1-psi^2)*(-2*eta);
      NJpsi=[NJpsi1,NJpsi2,NJpsi3,NJpsi4,NJpsi5,NJpsi6,NJpsi7,NJpsi8,NJpsi9];
      NJeta=[NJeta1,NJeta2,NJeta3,NJeta4,NJeta5,NJeta6,NJeta7,NJeta8,NJeta9];
      xpsi=NJpsi*xe'; ypsi=NJpsi*ye'; xeta=NJeta*xe'; yeta=NJeta*ye';
      jcob=xpsi*yeta-xeta*ypsi;
      psiMone=psi-1; etaMone=eta-1;
      psiPone=psi+1; etaPone=eta+1;
      psi_SQ=psi*psi; eta_SQ=eta*eta;
      N(1)=.25*psiMone*etaMone*psi*eta;
      N(2)=.25*psiPone*etaMone*psi*eta;
      N(3)=.25*psiPone*etaPone*psi*eta;
      N(4)=.25*psiMone*etaPone*psi*eta;
      N(5)= .5*etaMone*eta*(1.0-psi_SQ);
      N(6)= .5*psiPone*psi*(1.0-eta_SQ);
      N(7)= .5*etaPone*eta*(1.0-psi_SQ);
      N(8)= .5*psiMone*psi*(1.0-eta_SQ);
      N(9)=(1-psi_SQ)*(1-eta_SQ);
      xk=N*xe';yk=N*ye';
% exact displacements
      dispexactx=-P*yk/6/ebar/I*((6*L-3*xk)*xk+(2+nubar)*(yk*yk-D*D/4));
      dispexacty=P/6/ebar/I*(3*nubar*yk*yk*(L-xk)+D*D/4*(4+5*nubar)*xk+3*(L-xk/3)*xk*xk);

      errorx=(N*dispx')-dispexactx;
      errory=(N*dispy')-dispexacty;

      ergaus=((errorx)^2)+((errory)^2)*weight(i)*weight(j);
      erre=erre+ergaus*jcob;
end
end
end

