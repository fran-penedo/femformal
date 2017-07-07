%file shapef4_9.m
%forming shape functions for 9-node element

function [Nshap]=shapef4_9(psi,eta);
bkk=[1,3,3,1,2,3,2,1,2];
ckk=[1,1,3,3,1,2,3,2,2];

Npsi=[.5*psi*(psi-1),1-psi*psi,.5*psi*(psi+1)];
Neta=[.5*eta*(eta-1),1-eta*eta,.5*eta*(eta+1)];

for i=1:9
Nshap(i)=Npsi(bkk(i))*Neta(ckk(i));
end





