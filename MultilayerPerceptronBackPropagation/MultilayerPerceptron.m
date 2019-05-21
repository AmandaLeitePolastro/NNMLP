
lr = 0.1; %learning rate
dimlayers = [3,3,1]; %número de neurônios em cada camada (obs.: a última camada possui um neurônio porque a 
%saída só tem um valor)
nlayers = length(dimlayers); %número de camadas

%bias
B_layer = cell(1,nlayers);
B_layer{1} = [0.2, 0.2, 0.2];
B_layer{2} = [0.2, 0.2, 0.2];
B_layer{3} = [0.2];

%DATASET TREINO
n = 1; %número de vezes que o dataset de treino é apresentado para a rede no treino
data_treino = load('dados_class_treino.mat');
X_data_treino = data_treino.X'; %dataset de treino transposto
Y_real_treino = data_treino.s'; %label de treino transposto

%TREINO
W = MultilayerPerceptronBackPropagationTreino(X_data_treino, Y_real_treino, B_layer, dimlayers, lr, n);

%DATASET TESTE
data_teste = load('dados_class_teste.mat');
X_data_teste = data_teste.X'; %dataset de teste transposto
Y_real_teste = data_teste.s'; %label de teste transposto

%TESTE
YOutCalc = MultilayerPerceptronBackPropagationTeste(X_data_teste, W, B_layer, dimlayers);







