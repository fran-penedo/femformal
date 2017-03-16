function [ out ] = check_inv_rect_lp( A, b, R )

out = 1;

for j = 1:length(R)
    j
    A_j = A(j,[1:j-1, j+1:end])
    bb = -b(j) - A(j, j) * R(j,2)
    R_z = R([1:j-1, j+1:end],:)
    check = rect_in_semispc(R_z, A_j, bb);
    if ~check
        out = 0;
        break;
    end
    bb = -b(j) - A(j, j) * R(j,1);
    check = rect_in_semispc(R_z, -A_j, -bb);
    if ~check
        out = 0;
        break;
    end
end

end