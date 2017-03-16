function M = femass(xq,wq)

global nn ne xn conn h rho;

M = zeros(2,2);
Jac = h/2;  

for iq = 1:length(xq)
    M = M...
            + rho * Nfem(xq(iq))' * Nfem(xq(iq)) * wq(iq) * Jac;
end