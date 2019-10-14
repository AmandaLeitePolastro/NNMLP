
function [XShift] = ShiftEntradas(X, wind)

    %matriz de dados de entrada
    XVector = X'; %transforma a matriz de entrada em vetor linha

    %matriz final de dados de entrada
    %acrescenta a primeira linha da matriz de dados de entrada
    XShift = [0 0 XVector(1); 0 XVector(1) XVector(2)];

    for klinha = 1:(length(XVector) - (wind - 1))
        XShiftLinha = XVector(klinha:klinha + (wind - 1));
        XShift = [XShift; XShiftLinha]; %concatena as linhas formando uma matriz
    end
    
end
