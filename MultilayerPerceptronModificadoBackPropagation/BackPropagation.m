function e,w = BackPropagation(w_out, Ylayer, erro_out, xrow, lr)
    
    a = 1
    b = 1
    c = b/a
    
    DeltaOut = zeros(1,Ylayer) %delta de cada neuronio da última camada
    DeltaHL = zeros(1,)
    
    %calcula o delta de cada neurônio da última camada
    for kn = 1: length(Ylayer)
        erron = erro_out(kn);
        wn = w_out(kn);
        vn = b/a*(a - Ylayer(kn))*(a + Ylayer(kn));
        Deltan = erron*wn*vn;
        
        DeltaOut(kn) = Deltan;
    end
    
    %calcula o delta de cada neurônio de uma hidden layer
    for knhl = 1:lenght()
        
        
        
        
        return w_update, erro_update