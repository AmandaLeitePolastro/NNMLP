
function [YlayerOut] = MultilayerPerceptronBackPropagationTeste(X_teste, WTreino, B_layer, dimlayers)
    
    %inputs
    nrowx_teste = size(X_teste,1); %número de linhas da matriz X

    %network
    nlayers = length(dimlayers); %número de camadas
    
    %inicialização do vetor que irá retornar os y finais
    YlayerOut = zeros(1,nrowx_teste);
    
    %itera pelas linhas da matriz de inputs X
    for kx = 1:nrowx_teste %linha da matriz X 
        
        %seleciona a linha da matriz de input
        rowx = X_teste(kx,:); 
        
        %cálculo do valor de Y para cada camada
        Ylayer = CalcY(rowx, WTreino, B_layer, nlayers, dimlayers); 
        
        %cálculo do valor final dos y
        YlayerOut(kx) = Ylayer{nlayers};
    
    end
    
    YlayerOut = YlayerOut';
end
        
      
      
            
            