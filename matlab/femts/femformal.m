
n = 50;
L = 50;
T = [10 100];

[M, K, F] = femsystem(n, L, T);
[A, b] = femstandard(M, K, F);
[tA, tb, T] = femjordan(A, b);

R = repmat([-10000 10000], n, 1);
[A_R, b_R] = rect_to_poly(R);

tA_R = A_R / T;
tb_R = b_R;

tR_check = inside_rect(tA_R, tb_R);

check = check_inv_rect(tA, tb, tR_check)

[tA_R_check, tb_R_check] = rect_to_poly(tR_check);
A_R_check = tA_R_check * T;
b_R_check = tb_R_check;