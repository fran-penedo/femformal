% File: beammsh9.m

function [coord,numnod,numele,node,tnd,bnd,lnd,rnd]=beammsh9(xl,xr,numelex,yl,yr,numeley);

nlink=9;

numx=2*numelex+1;
numy=2*numeley+1;

xcd = linspace(xl,xr,numx);             % size(xcd)=[1,numx],  a row vector
ycd = linspace(yl,yr,numy);             % size(ycd)=[1,numy],  a row vector
lxcd = length(xcd);
lycd = length(ycd);
numnod = lxcd*lycd;
[xx yy] = meshgrid(xcd,ycd);            % size(xx)=[numx,numy]
                                        % size(yy]=[numx,numy]

xxx=xx';
yyy=yy';
coord = [xxx(:)';yyy(:)'];

numele = numelex*numeley;
node = zeros(nlink,numele);

lay1=1;                         % first node # on the layer 1
lay2=lay1+lxcd;                 %                           2
lay3=lay2+lxcd;                 %                           3

for iey=1:numeley
        
   for iex=1:numelex  
     ie=(iey-1)*numelex+iex;
          
     i1 = lay1;
     i2 = lay1+2;
     i3 = lay3+2;
     i4 = lay3;
     i5 = lay1+1;
     i6 = lay2+2;
     i7 = lay3+1;
     i8 = lay2;
     i9 = lay2+1;  
  
     lay1=lay1+2;
     lay2=lay2+2;
     lay3=lay3+2;
  
     node(1:nlink,ie) = [i1;i2;i3;i4;i5;i6;i7;i8;i9];
  end  % endfor iex
  
  lay1=lay1+1+lxcd;
  lay2=lay2+1+lxcd;
  lay3=lay3+1+lxcd;
  
end  % endfor iey

tnd = (numnod-lxcd+1):1:numnod;
bnd = 1:lxcd;
lnd = 1:lxcd:(numnod-lxcd+1);
rnd = lxcd:lxcd:numnod;


