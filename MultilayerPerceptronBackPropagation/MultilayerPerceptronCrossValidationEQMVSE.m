
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
n = 100; %número de vezes que o dataset de treino é apresentado para a rede no treino
data_treino = load('dados_class_treino.mat');
X_data_treino = data_treino.X'; %dataset de treino transposto
Y_real_treino = data_treino.s'; %label de treino transposto

%geração do dataset de treino com crossvalidation
DataSet = [X_data_treino Y_real_treino];

%divisão do dataset de treino em estimation e validation para crossvalidation
[data_est, data_val] = SplitTrainDataSetCV(DataSet); % 80% estimation e 20% validation

%separação entre dados e label
x_data_est = data_est(:,1:3);
y_data_est = data_est(:,4);
x_data_val = data_val(:,1:3);
y_data_val = data_val(:,4);

%DATASET TESTE
data_teste = load('dados_class_teste.mat');
X_data_teste = data_teste.X'; %dataset de teste transposto
Y_real_teste = data_teste.s'; %label de teste transposto


%GERAÇÃO E PLOT VSE
[x_VSE, VSE, EQM] = MultilayerPerceptronBackPropagationEQMVSE(x_data_est, y_data_est, x_data_val, y_data_val, B_layer, dimlayers, lr, n);

plot(x_VSE, VSE, 'blue', x_VSE, EQM, 'black');
grid on;

load handel
sound(y,Fs)



