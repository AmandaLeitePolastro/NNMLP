
function [YlayerOut] = MultilayerPerceptronModificadoBackPropagationTeste(XY_teste, WTreino, B_layer, dimlayers)
    
    %inputs
    nrowxy_teste = size(XY_teste,1); %número de linhas da matriz de input

    %network
    nlayers = length(dimlayers); %número de camadas
    
    %inicialização do vetor que irá retornar os y finais
    YlayerOut = zeros(1,nrowxy_teste);
    
    %itera pelas linhas da matriz de inputs X
    for kxy = 1:nrowxy_teste %linha da matriz X 
        
        %seleciona a linha da matriz de input
        rowxy = XY_teste(kxy,:); 
        
        %cálculo do valor de Y para cada camada
        Ylayer = CalcY(rowxy, WTreino, B_layer, nlayers, dimlayers); 
        
        %cálculo do valor final dos y
        YlayerOut(kxy) = Ylayer{nlayers};
    
    end
    
    YlayerOut = YlayerOut';
end
