
function [Wlayer] = MultilayerPerceptronBackPropagationTreino(X, Y_true, B_layer, dimlayers, lr, num)

%inputs
ncolx = size(X,2); %número de colunas da matriz de inputs
nrowx = size(X,1); %número de linhas da matriz de inputs

%network
nlayers = length(dimlayers); %número de camadas

%inicialização dos pesos
%dict que irá guardar as matrizes de valores dos pesos iniciais (zerados) de cada camada
Wlayer = StartW(ncolx, dimlayers, nlayers);

for kt = 1:num %itera pelo número de vezes que a amostra será apresentada à rede
    
    %itera pelas linhas da matriz de inputs X
    for kx = 1:nrowx %linha da matriz X
        
        %seleciona a linha da matriz de input
        rowx = X(kx,:); 
        
        %cálculo do valor de Y para cada camada
        Ylayer = CalcY(rowx, Wlayer, B_layer, nlayers, dimlayers);
        
        %cálculo dos deltas
        Deltalayer = CalcDelta(Wlayer, Ylayer, Y_true(kx), nlayers);
        
        %atualização dos pesos
        Wlayer = WUpdate(Wlayer, Ylayer, Deltalayer, rowx, lr);
        
    end
    
end

end




