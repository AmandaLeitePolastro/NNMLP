
function [Wlayer, Ylayer] = MultilayerPerceptronBackPropagation(X, Y_true, B_layer, dimlayers, lr)
    
    %inputs
    ncolx = size(X,2); %número de colunas da matriz X
    nrowx = size(X,1); %número de linhas da matriz X
    
    %network
    nlayers = length(dimlayers); %número de camadas
    
    %inicialização dos pesos
    %dict que irá guardar as matrizes de valores dos pesos iniciais (zerados) de cada camada
    Wlayer = StartW(ncolx, dimlayers, nlayers); 
    
    %itera pelas linhas da matriz de inputs X
    for kx = 1:nrowx %linha da matriz X 
        
        %seleciona a linha da matriz de input
        rowx = X(kx,:); 
        
        %cálculo do valor de Y para cada camada
        Ylayer = CalcY(rowx, Wlayer, B_layer, nlayers, dimlayers); 
            
        %cálculo dos deltas
        Deltalayer = CalcDelta(Wlayer, Ylayer, Y_true, nlayers);
        
        %atualização dos pesos
        Wlayer = WUpdate(Wlayer, Ylayer, Deltalayer, rowx, lr);
    
    end
end
        
      
      
            
            