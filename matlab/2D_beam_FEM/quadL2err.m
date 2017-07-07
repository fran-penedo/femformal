% Plot L2 error of beam problem using 4-node quad elements

h4 = [16/4,16/8,16/16,16/32,16/64];   % For 4-node
h9 = [16/8,16/16,16/32,16/64];   % For 9-node
L24 = [1.8977e-5,6.5674e-6,1.9958e-6,5.8539e-7];    % For 4 node quad 5 x 5 integration, 2x1 aspect ratio
L24 = [1.9452e-05,6.5915e-06,1.9834e-06,5.7924e-07,1.6606e-07];   % 4 node quad, square aspect ratio
%L24 = [1.9472e-05,6.6512e-06,2.0081e-06,5.8714e-07,1.6843e-07,4.7578e-08];    % 4 node quad, 8 x 8 integration
h4r = [16/64,16/128,16/256,16/512];
L24r = [4.5446e-06,1.3983e-06,4.0862e-07,1.1640e-07];	% For 4-node large aspect ratio, 8 x 8 integration
L29 = [1.9537e-7,2.4201e-8,3.0153e-9,3.7631e-10];  % For 9-node - give rate of 3.15
h4 = log10(h4);
h9 = log10(h9);
L24 = log10(L24);
L24r = log10(L24r);
L29 = log10(L29);

slope41 = (L24(2)-L24(1))/(h4(2)-h4(1))
slope42 = (L24(3)-L24(2))/(h4(3)-h4(2))
slope43 = (L24(4)-L24(3))/(h4(4)-h4(3))
slope44 = (L24(5)-L24(4))/(h4(5)-h4(4))
%slope45 = (L24(6)-L24(5))/(h4(6)-h4(5))

figure (1)
plot(h4,L24,'k'); hold on;
plot(h4r,L24r,'r'); hold on;
plot(h9,L29,'b');
legend('4-node Quad (x/y=1)','4-node Quad (x/y=8)','9-node Serendipity');
xlabel('Log10(h)');ylabel('Log10(error)');