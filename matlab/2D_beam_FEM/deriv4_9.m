%file deriv4_9.m
%derivatives of shape functions

function [NJpsi,NJeta]=deriv4_9(psi,eta);
bkk=[1,3,3,1,2,3,2,1,2];
ckk=[1,1,3,3,1,2,3,2,2];

Npsi=[.5*psi*(psi-1),1-psi*psi,.5*psi*(psi+1)];
Neta=[.5*eta*(eta-1),1-eta*eta,.5*eta*(eta+1)];
Nppsi=[.5*(2*psi-1),-2*psi,.5*(2*psi+1)];
Npeta=[.5*(2*eta-1),-2*eta,.5*(2*eta+1)];

for i=1:9
NJpsi(i)=Nppsi(bkk(i))*Neta(ckk(i));
NJeta(i)=Npsi(bkk(i))*Npeta(ckk(i));
end




