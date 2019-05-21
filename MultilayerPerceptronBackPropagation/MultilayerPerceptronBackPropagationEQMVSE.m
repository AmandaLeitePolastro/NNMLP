
function [ab, VSE, EQM] = MultilayerPerceptronBackPropagationEQMVSE(X_treino, Y_real_treino, X_teste, Y_real_teste, B_layer, dimlayers, lr, num)

%inputs treino
ncolx_treino = size(X_treino,2); %número de colunas da matriz X_treino
nrowx_treino = size(X_treino,1); %número de linhas da matriz X_treino

%dimensão erro
%número de amostras para treinar
namostras = 5;
dim_erro = (nrowx_treino/namostras)*num;
ab = [1:1:dim_erro];
VSE = [];
EQM = [];

%network
nlayers = length(dimlayers); %número de camadas

%inicialização dos pesos
%dict que irá guardar as matrizes de valores dos pesos iniciais (zerados) de cada camada
Wlayer = StartW(ncolx_treino, dimlayers, nlayers);


for kt = 1:num %itera pelo número de vezes que a amostra será apresentada à rede
    
    %contador das linhas da matriz de treino
    cont_linha = 1;
    
    %enquanto a matriz de treino possuir amostras
    while (cont_linha < length(X_treino))
        
        %conta o número de amostras utilizadas até o teste
        cont_aux = 0;
        
        %treino
        while (cont_aux ~= 5)
            
            %incremento do contador
            cont_aux = cont_aux + 1;
            
            %seleciona a linha da matriz de input
            rowx = X_treino(cont_linha,:);
            
            %cálculo do valor de Y para cada camada
            Ylayer = CalcY(rowx, Wlayer, B_layer, nlayers, dimlayers);
            
            %cálculo dos deltas
            Deltalayer = CalcDelta(Wlayer, Ylayer, Y_real_treino(cont_linha), nlayers);
            
            %atualização dos pesos
            Wlayer = WUpdate(Wlayer, Ylayer, Deltalayer, rowx, lr);
            
            %incremento do contador das linhas
            cont_linha = cont_linha + 1;
            
        end
        
        Y_teste_obtido_VSE = MultilayerPerceptronBackPropagationTeste(X_teste, Wlayer, B_layer, dimlayers); %teste (curva azul)
        Y_teste_obtido_EQM = MultilayerPerceptronBackPropagationTeste(X_treino, Wlayer, B_layer, dimlayers); %teste (curva preta)
        
        %cálculo do VSE
        Erro_VSE = (Y_teste_obtido_VSE - Y_real_teste).^2;
        VSE_temp = (sum(Erro_VSE)/length(Erro_VSE));
        VSE = [VSE VSE_temp];
        
        %cálculo do EQM
        Erro_EQM = (Y_teste_obtido_EQM - Y_real_treino).^2;
        EQM_temp = (sum(Erro_EQM)/length(Erro_EQM));
        EQM = [EQM EQM_temp];
    
    end
end
end




