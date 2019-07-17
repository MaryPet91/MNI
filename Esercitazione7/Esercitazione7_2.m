P=[0 0; 1 3; 3 5; 5 12; 7 4];
plot(P(:,1),P(:,2),'*k'); %plot punti di controllo
hold on
plot(P(:,1),P(:,2),'b'); %plot poligono di controllo
%curva di Bezier
[B] = de_Casteljau(P,0.01);
plot(B(:,1),B(:,2),'r'); %plot curva di Bezièr
Pmezzi = P/2;
plot(Pmezzi(3,1),Pmezzi(3,2),'*m'); %plot punti di controllo alterati
plot(Pmezzi(:,1),Pmezzi(:,2),'c');  %plot poligono di controllo alterato
[Bmezzi] = de_Casteljau(Pmezzi,0.01);
plot(Bmezzi(:,1),Bmezzi(:,2),'g'); %plot curva di Bezièr scalata
legend('punti di controllo','poligono di controllo', 'curva di Bezièr',
        ' punti di controllo alterati','poligono di controllo alterato',
        'curva di Bezièr scalata');
legend('Location','northwest');
hold off