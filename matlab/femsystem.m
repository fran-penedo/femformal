function [ M, K, F ] = femsystem( n, L, T )
%FEMSYSTEM Summary of this function goes here
%   Detailed explanation goes here

l = L/n;
z = zeros(1, n-3);

M = toeplitz([5 0 z]) * (l / 6);
K = toeplitz([2 -1 z]) / l;
F = [T(1) z T(2)]' / l;

end

