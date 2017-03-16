function [ R ] = inside_rect( A, b )

[c, r] = chebycenter(A, b);
n = length(c);
R = [c - repmat(r, n, 1), c + repmat(r, n, 1)];

end