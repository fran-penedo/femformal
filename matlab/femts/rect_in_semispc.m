function [ check ] = rect_in_semispc( R, a_s, b_s )

if ~any(a_s)
    check = (0 <= b_s);
else
    [x, fval] = linprog(-a_s', a_s, b_s + 1, [], [], R(:,1), R(:,2));
    [-fval, b_s]

    check = (-fval <= b_s);

end