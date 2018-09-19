%--------------------------------------------------------------------
% file: prob1.m, Nov/2002, modified for ME 365 FEM in Mechanics 
% 2D FEM matlab code for 4 or 9-node element
% Application to a cantilever beam
%--------------------------------------------------------------------
%clear all;

nnode=input(' Nodes per element (4 or 9): ');
MaxNgaus=3;

        % geometry parameters
L=16.0; c=2.0; D = 2*c;
xl=0.0;
xr=L;
yl=0.0;
yr=c;
h=2*c;
Inertia=h^3/12;

numelex=input('numelex=');
numeley=input('numeley=');

        % materials parameters
YoungModule=1.e7;
PossionRatio=0.3;
nubar = PossionRatio/(1-PossionRatio);
ebar = YoungModule/(1-PossionRatio^2);

        % load parameters
%Q_force=-1;
Q_force = -1;
% mesh of the beam
if (nnode==4) 
   [coord,numnod,numele,node,tnd,bnd,lnd,rnd]=beammsh4(xl,xr,numelex,yl,yr,numeley)
elseif (nnode==9) 
   [coord,numnod,numele,node,tnd,bnd,lnd,rnd]=beammsh9(xl,xr,numelex,yl,yr,numeley);
end 

x=coord(1,:);
y=coord(2,:);

ndof=2;
numeqns=numnod*ndof;

% external force
force=zeros(1,numeqns);
% nodal force due to tractions on surfaces
force=traction(nnode,lnd,rnd,Inertia,L,c,Q_force,y)
  
% displacement boundary condition
ifix=[zeros(2,numnod)];
ifix(:,1)=1;
ifix(1,lnd(length(lnd)))=1;
ifix(1,bnd)=1;

% assemble bigk matrix        
young=YoungModule*ones(1,numele);
pr=PossionRatio*ones(1,numele);

% zero bigk matrix to prepare for assembly
bigk=sparse(numeqns,numeqns);
%bigk = zeros(numeqns,numeqns);

%
%  loop over elements
%

for e=1:numele
   
  if (nnode==4)   
     gauss = [-1/sqrt(3),1/sqrt(3)];
     weight = [1,1];
     %gauss = [0];
     %weight = [2];
     [ke] = elem1(node,x,y,gauss,weight,young,pr,e);
     nlink  = 4;
  elseif (nnode==9)
     gauss = [-sqrt(3/5),0,sqrt(3/5)];
     weight = [5/9,8/9,5/9];
     %gauss = [-1/sqrt(3),1/sqrt(3)];
     %weight = [1,1];
     [ke] = elem4_9(node,x,y,gauss,weight,young,pr,e);
     nlink  = 9;
  end 


%
% assemble ke into bigk
%
   n1=ndof-1;
   for i=1:nlink;
     for j=1:nlink;
        if ( (node(i,e) ~= 0 )  & (node(j,e) ~= 0) )
          rbk=ndof*(node(i,e)-1)+1;
          cbk=ndof*(node(j,e)-1)+1;
          re=ndof*(i-1)+1;
          ce=ndof*(j-1)+1;  
          bigk(rbk:rbk+n1,cbk:cbk+n1)=bigk(rbk:rbk+n1,cbk:cbk+n1)+ke(re:re+n1,ce:ce+n1);
       end  % endif
     end;  % endfor j
   end;  % endfor i
end  % endfor e

% support conditions (boundary conditions)
for n=1:numnod
     for j=1:ndof
         if (ifix(j,n) == 1)
         m=ndof*(n-1)+j;
         bigk(m,:)=zeros(1,numeqns);
         bigk(:,m)=zeros(numeqns,1);
         bigk(m,m)=1.0;
         force(m)=0;
         end
    end
end

% solve stiffness equations
disp=force/bigk;

% display the mesh & displacement figure (to show the deformation)
figure(1);
dispplot;    

% display the comparison of exact & FEM displ solns of the middle surface
xx=[xl:(xr-xl)/100:xr];
xbar=L-xx;
ExactDispCoordX=xx;
v=nubar;
u2=xbar.^3 - L^3 -((4+5*v)*c*c + 3*L^2)*(xbar-L);
ExactDispY=Q_force/(6*ebar*Inertia)*u2;

figure(2);
plot(x(bnd),disp(2.*bnd),'-',ExactDispCoordX,ExactDispY,'-.');
title(['Comparison of Exact and FEM Displ Solns--mesh=',int2str(numelex),'*',int2str(numeley)]);
legend('FEM  Soln','Exact Soln');
xlabel('X');
ylabel('Displacement Component --- Uy');

% Compute energy norm
disperr = 0;
solerr = 0;
for e = 1:numele
    [erre]=dispnrm(node,nnode,x,y,e,disp,Q_force,ebar,nubar,Inertia,D,L);
    disperr = disperr + erre;
end
L2 = sqrt(disperr)