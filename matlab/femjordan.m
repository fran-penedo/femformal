function [ tA, tb, T ] = femjordan( A, b )
%FEMJORDAN Summary of this function goes here
%   Detailed explanation goes here

[T, tA] = eig(A);
tb = T*b;

end

