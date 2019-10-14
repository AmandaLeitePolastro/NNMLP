function W_layer = WUpdate(W_layer, Y_layer, Delta_layer, rowxy, lr)

for kl = length(Y_layer):-1:1 %percorre cada camada começando pela outputlayer
    Wlayer = W_layer{kl}; %seleciona a matriz de pesos da camada
    
    
    if(kl == 1) %é a primeira camada de todas (usa o x multiplicado por w para obter o y)
        
        %percorre cada neurônio de cada camada e atualiza a matriz de pesos
        for ncoluna = 1:size(Wlayer,2) %número de colunas
            for nlinha = 1:size(Wlayer,1) %número de linhas
                W_layer{kl}(nlinha,ncoluna) = Wlayer(nlinha,ncoluna) + lr*Delta_layer{kl}(ncoluna)*rowxy(nlinha);
            end
        end
        
    else %demais camadas (usa o vetor de y da camada anterior)
        for ncoluna = 1:size(Wlayer,2) %número de colunas
            for nlinha = 1:size(Wlayer,1)%número de linhas
                W_layer{kl}(nlinha,ncoluna) = Wlayer(nlinha,ncoluna) + lr*Delta_layer{kl}(ncoluna)*Y_layer{kl-1}(nlinha);
            end
        end
        
        
    end
    
end

end