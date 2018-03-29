import numpy as np

def BinaryProduct(X,Y):

    """ Binary Matrices or Matrix-vector product in Z/2Z. Works with scipy.sparse.csr_matrix matrices X,Y too."""

    A = X.dot(Y)

    return A%2

def InCode(H,x):

    """ Computes Binary Product of H and x. If product is null, x is in the code.
        Returns appartenance boolean.
    """

    return  (BinaryProduct(H,x)==0).all()

#pythran export Decoding_logBP(int[:,:], int list list, int list list, float[:,:], float[:], int)
def Decoding_logBP(H,Bits,Nodes,Lq,Lc,max_iter):
    """ Decoding function using Belief Propagation algorithm (logarithmic version)
    IMPORTANT: if H is large (n>1000), H should be scipy.sparse.csr_matrix object to speed up calculations
    (highly recommanded. )
    -----------------------------------
    Parameters:

    H: 2D-array (OR scipy.sparse.csr_matrix object) Parity check matrix, shape = (m,n)
    y: n-vector recieved after transmission in the channel. (In general, returned
    by Coding Function)
    Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN used in coding.

    max_iter: (default = 1) max iterations of the main loop. Increase if decoding is not error-free.
     """

    m,n=H.shape

    if not len(Lc)==n:
        raise ValueError('La taille de y doit correspondre au nombre de colonnes de H')

    if m>=n:
        raise ValueError('H doit avoir plus de colonnes que de lignes')


    # var = 10**(-SNR/10)

    # ### ETAPE 0: initialisation

    # Lc = 2*y/var

    # Lq=np.zeros(shape=(m,n))

    Lr = np.zeros(shape=(m,n))

    count=0

    prod=np.prod
    tanh = np.tanh
    log = np.log

    Lq += Lc

    #Bits,Nodes = BitsAndNodes(H)
    while(True):

        count+=1 #Compteur qui empêche la boucle d'être infinie ..

        #### ETAPE 1 : Horizontale
        for i in range(m):
            Ni = Bits[i]
            for j in Ni:
                Nij = Ni[:]
                Nij.remove(j)
                X = prod(tanh(0.5*Lq[i,Nij]))
                num = 1 + X
                denom = 1 - X
                # if num == 0:
                #     Lr[i,j] = -1
                # elif denom  == 0:
                #     Lr[i,j] =  1
                # else:
                #     Lr[i,j] = log(num/denom)
                Lr[i,j] = log(num) - log(denom)
        Lr = np.clip (Lr, -100, 100)

        #### ETAPE 2 : Verticale
        for j in range(n):
            Mj = Nodes[j]

            for i in Mj:
                Mji = Mj[:]
                Mji.remove(i)

                Lq[i,j] = Lc[j]+sum(Lr[Mji,j])


        #### LLR a posteriori:
        # L_posteriori = np.zeros(n)
        # for j in range(n):
        #     Mj = Nodes[j]

        #     L_posteriori[j] = Lc[j] + sum(Lr[Mj,j])
        extrinsic = np.empty(n)
        for j in range(n):
            Mj = Nodes[j]

            extrinsic[j] = sum(Lr[Mj,j])

        L_posteriori = extrinsic + Lc
        #x = np.array(L_posteriori <= 0).astype(int)
        x = np.array(extrinsic <= 0).astype(int)
        product = InCode(H,x)
        #print (count, product)

        if product or count >= max_iter:
            break
    # print (count)
    return np.array(L_posteriori <= 0).astype(int), Lq - Lc, extrinsic, product
