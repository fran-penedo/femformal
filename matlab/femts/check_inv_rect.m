function [ out ] = check_inv_rect( A, b, R )

check = [-A * R(:,1) - b; A * R(:,2) + b];
out = all(check <= 0);

end

