function Ylayer = CalcYTeste(xrow, WTreino, B_layer, nlayers, dimlayers)

Ylayer = cell(1,nlayers);

for kl = 1:length(dimlayers) %percorre cada camada
    
    Wlayer = WTreino{kl}; %seleciona a matriz de pesos calculada na aetapa de treino de cada camada
    
    if (kl ~= 1) %hidden layer
        xrow = Ylayer{kl-1};
    end
    
    Ylayer{kl} = tanh(xrow*Wlayer + B_layer{kl});
    
end

end