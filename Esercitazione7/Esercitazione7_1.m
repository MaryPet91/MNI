P=[2.5 -5; 9 9; 19 -2.5; 22.5 7.5];
plot(P(:,1),P(:,2),'*k'); %plot punti di controllo
hold on
plot(P(:,1),P(:,2),'b'); %plot poligono di controllo
%curva di Bezier
[B] = de_Casteljau(P,0.01);
plot(B(:,1),B(:,2),'r'); %plot curva di Bezièr
P(3,2)=-6;
plot(P(3,1),P(3,2),'*m'); %plot nuovo P2
plot(P(:,1),P(:,2),'c'); %plot nuovo poligono di controllo
[B] = de_Casteljau(P,0.01);
plot(B(:,1),B(:,2),'g'); %plot nuova curva di Bezièr 
legend('punti di controllo','poligono di controllo', 'curva di Bezièr',
        'nuovo P2','nuovo poligono di controllo','nuova curva di Bezièr');
legend('Location','south');
hold off