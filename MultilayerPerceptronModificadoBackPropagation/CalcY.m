function Ylayer = CalcY(xrow, W_layer, B_layer, nlayers, dimlayers)

Ylayer = cell(1,nlayers);

for kl = 1:length(dimlayers) %percorre cada camada

    Wlayer = W_layer{kl}; %seleciona a matriz de pesos de cada camada

    if (kl ~= 1) %hidden layer
        xrow = Ylayer{kl-1};
    end

    Ylayer{kl} = tanh(xrow*Wlayer + B_layer{kl});

end

end
