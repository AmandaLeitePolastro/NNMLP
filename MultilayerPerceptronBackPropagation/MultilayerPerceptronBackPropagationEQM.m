
function [Wlayer, ab_kx, EQM] = MultilayerPerceptronBackPropagationEQM(X_treino, Y_treino, X_teste, Y_real_teste, B_layer, dimlayers, lr, num)

%inputs treino
ncolx_treino = size(X_treino,2); %número de colunas da matriz X_treino
nrowx_treino = size(X_treino,1); %número de linhas da matriz X_treino

%EQM
EQM = zeros(1,nrowx_treino); %vetor que armazenará o EQM de cada amostra 
ab_kx = zeros(1,nrowx_treino); %vetor para plotar

%network
nlayers = length(dimlayers); %número de camadas

%inicialização dos pesos
%dict que irá guardar as matrizes de valores dos pesos iniciais (zerados) de cada camada
Wlayer = StartW(ncolx_treino, dimlayers, nlayers);


for kt = 1:num %itera pelo número de vezes que a amostra será apresentada à rede
    
    %itera pelas linhas da matriz de inputs X_treino
    for kx = 1:nrowx_treino %linha da matriz X_treino
        
        %abcissa para plotar
        ab_kx(kx) = kx;
        
        %seleciona a linha da matriz de input
        rowx = X_treino(kx,:); 
        
        %cálculo do valor de Y para cada camada
        Ylayer = CalcY(rowx, Wlayer, B_layer, nlayers, dimlayers);
        
        %cálculo dos deltas
        Deltalayer = CalcDelta(Wlayer, Ylayer, Y_treino(kx), nlayers);
        
        %atualização dos pesos
        Wlayer = WUpdate(Wlayer, Ylayer, Deltalayer, rowx, lr);
        
        %teste
        Y_obtido = MultilayerPerceptronBackPropagationTeste(X_teste, Wlayer, B_layer, dimlayers);
        
        %cálculo do EQM
        Erro = (Y_teste_obtido - Y_real_teste').^2;
        EQM_kx = sum(Erro)/length(Erro);
        EQM(kx) = EQM_kx;
    
    end
    
    
    
    
    
end

end




