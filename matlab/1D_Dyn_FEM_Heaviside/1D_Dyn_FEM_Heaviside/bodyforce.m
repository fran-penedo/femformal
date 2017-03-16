% Subroutine to generate element body force

function f = bodyforce(b,xq,wq)
global nn ne xn conn h E;

f = zeros(2,1);
Jac = h/2;                     % Jacobian

for i = 1:length(xq)
    f = f + Nfem(xq(i))' * b * wq(i) * Jac;
end