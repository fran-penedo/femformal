function [ A, b ] = femstandard( M, K, F )

Ab = M\[-K F];
A = Ab(:,1:end-1);
b = Ab(:,end);

end

