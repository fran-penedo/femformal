% Calculate element stiffness matrix

function K = festiff(xq,wq)

global nn ne xn conn h rho E;

K = zeros(2,2);

iJac = 2/h;                      % Element jacobian

for iq = 1:length(xq)
    K = K...
            + E * dNfem(xq(iq))' * dNfem(xq(iq)) * wq(iq) * iJac;
end