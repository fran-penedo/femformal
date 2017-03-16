function N = Nfem(x)
% 1D FEM shape function

global nn ne xn conn h;

N = [.5*(1-x) .5*(1+x)];    % FEM quadratic shape functions in 1D isoparametric parent domain
