
function [Wlayer] = MultilayerPerceptronModificadoBackPropagationTreino(XY, Y_true, B_layer, dimlayers, lr, num, order)

%inputs
ncolx = size(XY,2) + order ; %número de colunas da matriz X + número de amostras de y a serem comnsideradas   
nrowxy = size(XY,1); %número de linhas da matriz X

%network
nlayers = length(dimlayers); %número de camadas

%inicialização dos pesos
%dict que irá guardar as matrizes de valores dos pesos iniciais (zerados) de cada camada
Wlayer = StartW(ncolx, dimlayers, nlayers);


for kt = 1:num %itera pelo número de vezes que a amostra será apresentada à rede
    
    %itera pelas linhas da matriz de inputs X
    for kxy = 1:nrowxy %linha da matriz XY
        
        %seleciona a linha da matriz de input
        rowxy = XY(kxy,:); 
        
        %cálculo do valor de Y para cada camada
        Ylayer = CalcY(rowxy, Wlayer, B_layer, nlayers, dimlayers); %ok
        
        %cálculo dos deltas
        Deltalayer = CalcDelta(Wlayer, Ylayer, Y_true(kxy), nlayers); %ok
        
        %atualização dos pesos
        Wlayer = WUpdate(Wlayer, Ylayer, Deltalayer, rowxy, lr); %ok
        
    end
    
end

end




