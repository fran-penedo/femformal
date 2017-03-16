function [ A, b ] = rect_to_poly( R )

n = length(R);
A = [eye(n); -eye(n)];
b = [R(:,1); R(:,2)];

end

