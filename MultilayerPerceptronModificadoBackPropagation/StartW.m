function [Wlayer] = StartW(ncolx, dimlayers, nlayers)
    
Wlayer = cell(1,nlayers);
 
    for kl = 1:nlayers %percorre cada camada
        
        if(kl == 1) %se é a primeira camada (utiliza as dimensões da matriz x)
            Wlayer{kl} = zeros(ncolx, dimlayers(kl));
        
        else %se não é a primeira camada (utiliza o y como x e por isso o número de colunas do vetor de y anterior também pode 
             %ser determinado pela matriz de pesos da camada anterior
            Wlayer{kl} = zeros(dimlayers(kl-1), dimlayers(kl)); %cada elemento do array é a matriz de pesos e cada camada
            
        end
    end 
end 
