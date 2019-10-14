
lr = 0.1; %learning rate
dimlayers = [2,1]; %número de neurônios em cada camada (obs.: a última camada possui um neurônio porque a 
%saída só tem um valor)
nlayers = length(dimlayers); %número de camadas
order = 2; %ordem do filtro FIR utilizado
wind = 3; %número de atributos da matriz de inputs

%bias
B_layer = cell(1,nlayers);
B_layer{1} = [0.2, 0.2, 0.2];
B_layer{2} = [0.2, 0.2, 0.2];
B_layer{3} = [0.2];

%DATASET TREINO
n = 1; %número de vezes que o dataset de treino é apresentado para a rede no treino
data_treino = load('dados_class_treino.mat');

%BASE DE TREINO COM SHIFT
X_data_treino_temp = data_treino.X'; %dataset de treino transposto
X_data_coluna = X_data_treino_temp(:,1); %pega a primeira coluna da matriz de inputs porque ela tem os atributos x1, x2, x3...
X_data_treino = ShiftEntradas(X_data_coluna, wind); %essa matriz é gerada somente por um atributo (primeira coluna da matriz X_data_treino_temp.
Y_real_treino = data_treino.s'; %label de treino transposto

%TREINO
XY_input_treino = GeradorMatrizEntrada(X_data_treino, Y_real_treino, order);
W_treino = MultilayerPerceptronModificadoBackPropagationTreino(XY_input_treino, Y_real_treino, B_layer, dimlayers, lr, n, order);

%DATASET TESTE
data_teste = load('dados_class_teste.mat');
X_data_teste_temp = data_teste.X'; %dataset de teste transposto
X_data_teste = ShiftEntradas(X_data_teste_temp); %dataset de teste transposto
Y_real_teste = data_teste.s'; %label de teste transposto

%TESTE
XY_input_teste = GeradorMatrizEntrada(X_data_teste, Y_real_teste, order);
YOutCalc = MultilayerPerceptronModificadoBackPropagationTeste(XY_input_teste, W_treino, B_layer, dimlayers);







