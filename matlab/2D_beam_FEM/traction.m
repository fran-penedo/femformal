function force=traction(nnode,lnd,rnd,I,L,c,P,y)
% this function calculate nodal forces due to traction BC's
% written by SW
force=zeros(1,2*length(y));
for jside=1:2  % 1 is left side while 2 is rightside
  if jside==1
     jend=lnd;
  else
     jend=rnd;
  end
  xfrc=2*jend-ones(1,length(jend));
  yfrc=2*jend;
  if nnode==4  
    numeley=length(jend)-1;
  else
    numeley=(length(jend)-1)/2;
  end   
  for le=1:numeley
    if nnode==4
      y_le=[y(jend(le)) y(jend(le+1))]';
      f_le=getf_4_9(nnode,jside,I,L,c,P,y_le);
      force(xfrc(le))=force(xfrc(le))+f_le(1);
      force(yfrc(le))=force(yfrc(le))+f_le(2);   
      force(xfrc(le+1))=force(xfrc(le+1))+f_le(3);
      force(yfrc(le+1))=force(yfrc(le+1))+f_le(4);    
    elseif nnode==9
      y_le=[y(jend(2*le-1)) y(jend(2*le))  y(jend(2*le+1))]';
      f_le=getf_4_9(nnode,jside,I,L,c,P,y_le);
      force(xfrc(2*le-1))=force(xfrc(2*le-1))+f_le(1);
      force(yfrc(2*le-1))=force(yfrc(2*le-1))+f_le(2);   
      force(xfrc(2*le))=force(xfrc(2*le))+f_le(3);
      force(yfrc(2*le))=force(yfrc(2*le))+f_le(4);    
      force(xfrc(2*le+1))=force(xfrc(2*le+1))+f_le(5);
      force(yfrc(2*le+1))=force(yfrc(2*le+1))+f_le(6);    
    else
      stop;
    end %end if
  end % end of le elelment loop on one side
end % end of side loop

%-------------------------------------------------------------
%begin of element traction bound nodal force vectors function 

function F=getf_4_9(nnode,lft_rt,I,L,c,P,y)
%This function computes and returns the total nodal force vectors
%Gauss integration method is used
% if lft_rt=1 it's the left side, otherwise the right side

ksi_2=[-1 1]/sqrt(3.0); 
ksi_3=[-1 0 1]*sqrt(3.0/5.0); 

if nnode==4 
    F=zeros(4,1);
    le=y(2)-y(1);
   %2-point Gauss quadrature
    N1_gauss=N1(ksi_2);
    N2_gauss=N2(ksi_2);
    y1=[N1_gauss(1) N2_gauss(1)]*y;
    y2=[N1_gauss(2) N2_gauss(2)]*y;
    if lft_rt==1    % left side
      tmp1=N1_gauss(1)*trac_lft(I,L,c,P,y1) + N1_gauss(2)*trac_lft(I,L,c,P,y2);
      tmp2=N2_gauss(1)*trac_lft(I,L,c,P,y1) + N2_gauss(2)*trac_lft(I,L,c,P,y2);
      F(1:2)=tmp1*le/2;
      F(3:4)=tmp2*le/2;
    elseif lft_rt==2  % right side
      tmp1=N1_gauss(1)*trac_rt(I,L,c,P,y1) + N1_gauss(2)*trac_rt(I,L,c,P,y2);
      tmp2=N2_gauss(1)*trac_rt(I,L,c,P,y1) + N2_gauss(2)*trac_rt(I,L,c,P,y2);
      F(2)=tmp1(2)*le/2;
      F(4)=tmp2(2)*le/2;
    else
      stop;
    end
elseif nnode==9
    F=zeros(6,1);
    le=y(3)-y(1);
   %3-point Gauss quadrature
    N1_gauss=NQu1(ksi_3);
    N2_gauss=NQu2(ksi_3);
    N3_gauss=NQu3(ksi_3);
    y1=[N1_gauss(1) N2_gauss(1) N3_gauss(1)]*y;
    y2=[N1_gauss(2) N2_gauss(2) N3_gauss(2)]*y;
    y3=[N1_gauss(3) N2_gauss(3) N3_gauss(3)]*y;   
    if lft_rt==1    % left side
      h1=trac_lft(I,L,c,P,y1);
      h2=trac_lft(I,L,c,P,y2);
      h3=trac_lft(I,L,c,P,y3);      
      tmp1=5/9*N1_gauss(1)*h1 + 8/9*N1_gauss(2)*h2 + 5/9*N1_gauss(3)*h3;
      tmp2=5/9*N2_gauss(1)*h1 + 8/9*N2_gauss(2)*h2 + 5/9*N2_gauss(3)*h3;
      tmp3=5/9*N3_gauss(1)*h1 + 8/9*N3_gauss(2)*h2 + 5/9*N3_gauss(3)*h3;
      F(1:2)=tmp1*le/2;
      F(3:4)=tmp2*le/2;
      F(5:6)=tmp3*le/2;
    elseif lft_rt==2  % right side
      h1=trac_rt(I,L,c,P,y1);
      h2=trac_rt(I,L,c,P,y2);
      h3=trac_rt(I,L,c,P,y3);      
      tmp1=5/9*N1_gauss(1)*h1 + 8/9*N1_gauss(2)*h2 + 5/9*N1_gauss(3)*h3;
      tmp2=5/9*N2_gauss(1)*h1 + 8/9*N2_gauss(2)*h2 + 5/9*N2_gauss(3)*h3;
      tmp3=5/9*N3_gauss(1)*h1 + 8/9*N3_gauss(2)*h2 + 5/9*N3_gauss(3)*h3;
      F(1:2)=tmp1*le/2;
      F(3:4)=tmp2*le/2;
      F(5:6)=tmp3*le/2;
    else
      stop;
    end
else
  stop;
end
% end of getf_4_9.m 


%--------------------------------------------
% funcions describing the traction forces
         
function trac = trac_lft(I,L,c,P,y)
         trac = zeros(2,1);
         trac(1)=P*L*y/I;
         trac(2)=-P*(c*c-y*y)/(2*I);

function trac=trac_rt(I,L,c,P,y)
         trac=zeros(2,1);
         trac(2)= P*(c*c-y*y)/(2*I);


%--------------------------------------------
%1-D linear and quadratic shape functions

function nfun=N1(ksi)
     nfun=0.5*(1-ksi);
     
function nfun=N2(ksi)
     nfun=0.5*(1+ksi);

function nfun=NQu1(ksi)
     nfun=0.5*ksi.*(ksi-1);
     
function nfun=NQu2(ksi)
     nfun=1.0-ksi.*ksi;

function nfun=NQu3(ksi)
     nfun=0.5*ksi.*(ksi+1);

%-------------------------------------------------------------     

