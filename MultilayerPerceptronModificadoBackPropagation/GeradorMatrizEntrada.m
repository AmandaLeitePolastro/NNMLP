function [XY] = GeradorMatrizEntrada(X, Y_true, order)

    XY = [];
    nrowx = size(X,1); %número de linhas da matriz X

    %número de amostras 0 anteriores ao primeiro valor de y 
    amostras = zeros(1, order);  

    %transforma a coluna dos labels em vetor e acrescenta mais 3 amostras zero no ínicio do vetor
    Y_true_row = [amostras Y_true'];
    
    %itera pelas linhas da matriz de inputs X
    for kx = 1:nrowx %linha da matriz X
        
        %seleciona a linha da matriz de input
        rowx = X(kx,:); 
        
        %seleciona as amostras de y (amostras y que serão misturadas às
        %amostras x)
        rowy = Y_true_row(kx:kx+1);
        
        %concatena as amostras x e y
        rowxy = [rowy rowx];
      
    end
        XY = vertcat(XY,rowxy);
    
end
