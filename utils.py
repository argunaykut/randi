
import numpy as np


def data_norm(traj_set,dim,task,thr=1e-10):  
    
    '''function to normalize a set of trajectories of the same length l.
    takes as input a vector of length N l*dim 
    for segmenation task, set task=3. It returns array of normalized displacements of dimension (N,dim,l-1)
    for other tasks, it returns array of normalized displacements of dimension (N,dim,l) where the last entries are all 0

    '''
        
    N = len(traj_set)
    r = np.array(traj_set).reshape(N,dim,-1)  
    r_3 = np.copy(r)
    r = np.diff(r,axis=2)                              # get the increments

    for dm in range(dim):
        x = np.copy(r[:,dm,:])                                     # get x data
        sx = np.std(x,axis=1)                           
        x = (x-np.mean(x,axis=1).reshape(len(x),1)) / np.where(sx>thr,sx,1).reshape(len(x),1)   # normalize x data
        if task == 3:
            x =  np.concatenate((x,np.zeros((N,1))),axis=1)         #if the task is 3, each dimension of the trajectory gets a 0 at the end

            r_3[:,dm,:]  = np.copy(x)
        else:    
            r[:,dm,:]  = np.copy(x)


    if task == 3:
        
        return r_3
    else:
        return r
    
    
    
def data_reshape(r,bs,dim):  
    
    '''function to prepare a set of trajectories of the same length into
    the shape required by the network. bs is the block size. 
    takes as input array of normalized displacements of dimension (N,dim,js)
    The function automatically cuts the trajectory to
    the largest multiple of bs. The reshaping e.g. for a 2-dimensional trajectory
    for a net working on blocks of dimension 4 gives the trajectory reshaped as
    { [x0,y0, x1, y1], [x2,y2, x3,y3], ...} '''
            
    js = r.shape[-1]
    N = r.shape[0]


    rl=int(dim*(js)/bs)*int(bs/dim)  #cutting the trajectory to fit to  multiple of dimensione used by net


    rt = np.transpose(r[:,:,:rl],axes = [0,2,1])
#         print(rl, rt.shape)
    rs_traj = rt.reshape(N,-1,bs)
    
    return rs_traj

    
def many_net_uhd(nets,traj_set,centers,dim,task,thr=1e-10,skip=[]):
    """Function to apply a combination of the nearest nets to a set of trajectories of the same length
    Takes as input list of networks, data set and
    the vector centers of where the different nets
    were trained on. 
    The input trajectory is given by an array where the dimensions are concatenated:
        traj=(x_0...xN,y0,...yN)
    The 2-dimensional trajectory is reshaped according to the network dimension. 
    e.g. for a net working on blocks of dimension 4 the trajectory is reshaped as
    { [x0,y0, x1, y1], [x2,y2, x3,y3], ...}
    All tajectories need to have the same length
    """
    centers=np.asarray(centers)
    n_nets=len(nets) #number of nets we can use
   
    #obtaining the shape of the input required by each of the networks
    di=[]
    for n in nets:
        di.append(n.layers[0].input_shape[-1])
    di=np.asarray(di)
    

    X = np.asarray(traj_set)
    jj= X.shape[1]           #length of trajectory times dimension
    js=int(jj/dim)          #length of trajectory
    #choosing which net to use
    if js<=centers[0]:
        k=0
    elif js>np.max(centers):
        k=n_nets-1
    else:

        k=np.argmax(js<np.asarray(centers))-1


    #taking the diff and reshaping the trajectory
    r_norm = data_norm(traj_set,dim,task=task,thr=thr)
    rs_traj = data_reshape(r_norm,bs = di[k],dim = dim)
    
    pr_b=nets[k].predict(rs_traj).flatten()

    if ((k<n_nets-1) and np.isin(k,skip,invert=True) ):
        #distance between the net used and the following one
        ran=centers[k+1]-centers[k]
        d=(js-centers[k])/ran   #distance between traj len and center of net used
        if d>0:
            rs_traj_b = data_reshape(r_norm,bs = di[k+1],dim=dim)

            pr_2b=nets[k+1].predict(rs_traj_b).flatten()
            pr_b=((1-d)*pr_b+d*pr_2b)


    return np.asarray(pr_b).flatten()

def my_atan(x1,x2):
    '''function to compute the arctan'''
    y=np.arctan2(x1,x2)
    b=y<0
    c=b.astype(int)*(2*np.pi)
    d=y+c 
    return    d;   