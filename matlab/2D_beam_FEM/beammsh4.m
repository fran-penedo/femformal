% File: BeamMsh4.m

function [coord,nnds,ncells,conne,tnd,bnd,lnd,rnd]=beammsh4(xl,xr,numelex,yl,yr,numeley);

numx=numelex+1;
numy=numeley+1;

xcd = linspace(xl,xr,numx);
ycd = linspace(yl,yr,numy);
lxcd = length(xcd);
lycd = length(ycd);
nnds = lxcd*lycd;
[xx yy] = meshgrid(xcd,ycd);
xxx=xx';
yyy=yy';
coord = [xxx(:)';yyy(:)'];
ncells = (lxcd-1)*(lycd-1);

        % modified by XUE GANG.
        % to be compatible with the 9-node elements
%conne = zeros(4,ncells);
conne = zeros(9,ncells);

for i=1:(lycd-1)
  lay1 = (i-1)*lxcd;
  lay2 = i*lxcd;
  count = (i-1)*(lxcd-1);
  cell = (count+1):1:(count+lxcd-1);
  in1 = (lay1+1):1:(lay2-1);
  in2 = (lay1+2):1:lay2;
  in3 = (lay2+2):1:(lay2+lxcd);
  in4 = (lay2+1):1:(lay2+lxcd-1);
  conne(1:4,cell) = [in1;in2;in3;in4];
end  
tnd = (nnds-lxcd+1):1:nnds;
bnd = 1:lxcd;
lnd = 1:lxcd:(nnds-lxcd+1);
rnd = lxcd:lxcd:nnds;


