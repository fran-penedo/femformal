
n = 100;
L = 100;
T = [10 100];

[M, K, F] = femsystem(n, L, T);
[A, b] = femstandard(M, K, F)

R = repmat([5 105], n-1, 1);

check_inv_rect_lp(A, b, R)

