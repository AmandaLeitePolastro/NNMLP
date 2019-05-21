%calcula os deltas da output layer e das hidden layers na mesma function

function Delta = CalcDeltav2(W_layer, Y_layer, Y_true, nlayers)

a = 1;
b = 1;
c = b/a;

Delta = cell(1,nlayers);
erro_out = Y_true - Y_layer{nlayers}; 

%percorre cada camada começando pela outputlayer
for kl = length(Y_layer):-1:1 %percorre cada camada começando pela outputlayer
    
    Wlayer = W_layer{kl};
    
    Ylayer = Y_layer{kl};
    %Deltalayer = zeros(1,length(Ylayer));
    
    if(kl == nlayers) %é outputlayer
        Deltalayer = zeros(1,length(Ylayer));
        
        %percorre os neurônios de cada camada
        for kn = 1: size(Wlayer,2) %número de colunas de cada matriz de pesos
            vn = c*(a - Ylayer(kn))*(a + Ylayer(kn)); %calcula o vn para cada neurônio
            Deltalayer(kn) = erro_out(kn)*vn; %preenche o vetor com os deltas da output layer
        end
        
        Delta_prev = Deltalayer;
        
    else %é hidden layer
        
        %percorre os neurônios de cada camada
        for kn = 1: size(Wlayer,2) 
            vn = c*(a - Ylayer(kn))*(a + Ylayer(kn)); %calcula o vn para cada neurônio
            WlayerT = W_layer{kl+1}(kn,:)'; %linha da matriz de pesos da camada (coluna transposta)
            Deltalayer(kn) = vn*Delta_prev*WlayerT; %preenche o vetor com os deltas da hidden layer utilizando o vetor de
            %deltas da camada seguinte
        end
        
    Delta_prev = Deltalayer;
    
    end
    
    Delta{kl} = Deltalayer; %preenche o cell array com o delta de cada camada
    
end

end