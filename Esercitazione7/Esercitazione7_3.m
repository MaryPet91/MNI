t=[0;0;0;0;1;2;3;4;5;5;6;7];
P=[-2.5 -8; 0 3; 1 -2; 5 -7.5; 10 4; 11 6; 15 4; 22.5 7.5];
plot(P(:,1),P(:,2),'*k'); %plot punti di controllo
hold on
plot(P(:,1),P(:,2),'b'); %plot poligono di controllo
Pt = transpose(P);
n=4;
[C,U] = bspline_deboor(n,t,Pt);
plot(C(1,:),C(2,:),'r'); %plot B-spline
legend('punti di controllo','poligono di controllo','B-spline');
legend('Location','southeast');
hold off