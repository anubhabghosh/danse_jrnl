#####################################################
# Creators: Anubhab Ghosh, Antoine Honoré
# Feb 2023
#####################################################
from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tikzplotlib
import os

def plot_state_trajectory(X, X_est_KF=None, X_est_EKF=None, X_est_UKF=None, X_est_DANSE=None, X_est_KNET=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    #print(X.shape)
    
    if X.shape[-1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X[:,0], X[:,1],'k-',label='$\\mathbf{x}^{true}$')
        if not X_est_KF is None:
            ax.plot(X_est_KF[:,0], X_est_KF[:,1],':',label='$\\hat{\mathbf{x}}_{KF}$')
        if not X_est_EKF is None:
            ax.plot(X_est_EKF[:,0], X_est_EKF[:,1],'b.-',label='$\\hat{\mathbf{x}}_{EKF}$')
        if not X_est_UKF is None:
            ax.plot(X_est_UKF[:,0], X_est_UKF[:,1],'-.',label='$\\hat{\mathbf{x}}_{UKF}$')
        if not X_est_DANSE is None:
            ax.plot(X_est_DANSE[:,0], X_est_DANSE[:,1],'r--',label='$\\hat{\mathbf{x}}_{DANSE}$')
        if not X_est_KNET is None:
            ax.plot(X_est_KNET[:,0], X_est_KNET[:,1], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet}$')
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        plt.legend()
        
    elif X.shape[-1] > 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:,0], X[:,1], X[:,2], 'k-', label='$\\mathbf{x}^{true}$')
        if not X_est_KF is None:
            ax.plot(X_est_KF[:,0], X_est_KF[:,1], X_est_KF[:,2], ':',label='$\\hat{\mathbf{x}}_{KF}$')
        if not X_est_EKF is None:
            ax.plot(X_est_EKF[:,0], X_est_EKF[:,1], X_est_EKF[:,2], 'b.-', label='$\\hat{\mathbf{x}}_{EKF}$', lw=1.3)
        if not X_est_UKF is None:
            ax.plot(X_est_UKF[:,0], X_est_UKF[:,1], X_est_UKF[:,2], 'x-', ms=4, color="orange", label='$\\hat{\mathbf{x}}_{UKF}$', lw=1.3)
        if not X_est_KNET is None:
            ax.plot(X_est_KNET[:,0], X_est_KNET[:,1], X_est_KNET[:,2], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet}$', lw=1.3)
        if not X_est_DANSE is None:
            ax.plot(X_est_DANSE[:,0], X_est_DANSE[:,1], X_est_DANSE[:,2], 'r--',label='$\\hat{\mathbf{x}}_{DANSE}$', lw=1.3)
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        ax.set_zlabel('$X_3$')
        #handles, labels = ax.get_legend_handles_labels()
        #order=None
        #if order is None:
        #    order=range(len(handles))
        #ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],ncol=5,fontsize=10)
        #ax.get_legend().set_bbox_to_anchor(bbox=(1,0))
        plt.legend()
        plt.tight_layout()

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig_name)
        tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")
    #plt.show()
    return None

def plot_state_trajectory(X, X_est_KF=None, X_est_EKF=None, X_est_UKF=None, X_est_DANSE=None, X_est_KNET=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    #print(X.shape)
    
    if X.shape[-1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X[:,0], X[:,1],'k-',label='$\\mathbf{x}^{true}$')
        if not X_est_KF is None:
            ax.plot(X_est_KF[:,0], X_est_KF[:,1],':',label='$\\hat{\mathbf{x}}_{KF}$')
        if not X_est_EKF is None:
            ax.plot(X_est_EKF[:,0], X_est_EKF[:,1],'b.-',label='$\\hat{\mathbf{x}}_{EKF}$')
        if not X_est_UKF is None:
            ax.plot(X_est_UKF[:,0], X_est_UKF[:,1],'-.',label='$\\hat{\mathbf{x}}_{UKF}$')
        if not X_est_DANSE is None:
            ax.plot(X_est_DANSE[:,0], X_est_DANSE[:,1],'r--',label='$\\hat{\mathbf{x}}_{DANSE}$')
        if not X_est_KNET is None:
            ax.plot(X_est_KNET[:,0], X_est_KNET[:,1], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet}$')
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        plt.legend()
        
    elif X.shape[-1] > 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:,0], X[:,1], X[:,2], 'k-', label='$\\mathbf{x}^{true}$')
        if not X_est_KF is None:
            ax.plot(X_est_KF[:,0], X_est_KF[:,1], X_est_KF[:,2], ':',label='$\\hat{\mathbf{x}}_{KF}$')
        if not X_est_EKF is None:
            ax.plot(X_est_EKF[:,0], X_est_EKF[:,1], X_est_EKF[:,2], 'b.-', label='$\\hat{\mathbf{x}}_{EKF}$', lw=1.3)
        if not X_est_UKF is None:
            ax.plot(X_est_UKF[:,0], X_est_UKF[:,1], X_est_UKF[:,2], 'x-', ms=4, color="orange", label='$\\hat{\mathbf{x}}_{UKF}$', lw=1.3)
        if not X_est_KNET is None:
            ax.plot(X_est_KNET[:,0], X_est_KNET[:,1], X_est_KNET[:,2], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet}$', lw=1.3)
        if not X_est_DANSE is None:
            ax.plot(X_est_DANSE[:,0], X_est_DANSE[:,1], X_est_DANSE[:,2], 'r--',label='$\\hat{\mathbf{x}}_{DANSE}$', lw=1.3)
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        ax.set_zlabel('$X_3$')
        #handles, labels = ax.get_legend_handles_labels()
        #order=None
        #if order is None:
        #    order=range(len(handles))
        #ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],ncol=5,fontsize=10)
        #ax.get_legend().set_bbox_to_anchor(bbox=(1,0))
        plt.legend()
        plt.tight_layout()

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig_name)
        tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")
    #plt.show()
    return None

def plot_state_trajectory_w_lims(X, X_est_KF=None, X_est_KF_std=None, 
                                X_est_EKF=None, X_est_EKF_std=None, 
                                X_est_UKF=None, X_est_UKF_std=None, 
                                X_est_DANSE=None, X_est_DANSE_std=None,
                                X_est_DANSE_sup=None, X_est_DANSE_sup_std=None,
                                X_est_KNET=None, X_est_KNET_std=None,
                                savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    #print(X.shape)
    
    if X.shape[-1] == 2:
        T_start=33
        T_end=165#X.shape[0]
        idim=1
        lw=1.3
        #plt.rcParams['font.size'] = 16
        plt.rcParams['font.family']='serif'
        fig, ax = plt.subplots()
        #plt.subplot(311)
        if not X_est_UKF is None:
            ax.plot(X_est_UKF[T_start:T_end,idim], 'x-',ms=5,color="orange",label='$\\hat{\mathbf{x}}_{UKF}$',lw=lw)
            ax.fill_between(np.arange(X_est_UKF[T_start:T_end,idim].shape[0]), 
                            X_est_UKF[T_start:T_end,idim] - X_est_UKF_std[T_start:T_end,idim], 
                            X_est_UKF[T_start:T_end,idim] + X_est_UKF_std[T_start:T_end,idim],
                            facecolor='orange', alpha=0.4)
        if not X_est_DANSE is None:
            ax.plot(X_est_DANSE[T_start:T_end,idim], 'rs-',label='$\\hat{\mathbf{x}}_{DANSE}$',lw=lw, ms=2)
            ax.fill_between(np.arange(X_est_DANSE[T_start:T_end,idim].shape[0]), 
                            X_est_DANSE[T_start:T_end,idim] - X_est_DANSE_std[T_start:T_end,idim], 
                            X_est_DANSE[T_start:T_end,idim] + X_est_DANSE_std[T_start:T_end,idim],
                            facecolor='red', alpha=0.4)
        if not X_est_DANSE_sup is None:
            ax.plot(X_est_DANSE_sup[T_start:T_end,idim], 'm^-',label='$\\hat{\mathbf{x}}_{DANSE-Supervised}$',lw=lw, ms=3)
            ax.fill_between(np.arange(X_est_DANSE_sup[T_start:T_end,idim].shape[0]), 
                            X_est_DANSE_sup[T_start:T_end,idim] - X_est_DANSE_sup_std[T_start:T_end,idim], 
                            X_est_DANSE_sup[T_start:T_end,idim] + X_est_DANSE_sup_std[T_start:T_end,idim],
                            facecolor='magenta', alpha=0.4)
        if not X_est_KNET is None:
            plt.plot(X_est_KNET[T_start:T_end,idim], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet} $', lw=lw)
            ax.fill_between(np.arange(X_est_KNET[T_start:T_end,idim].shape[0]), 
                            X_est_KNET[T_start:T_end,idim] - X_est_KNET_std[T_start:T_end,idim], 
                            X_est_KNET[T_start:T_end,idim] + X_est_KNET_std[T_start:T_end,idim],
                            facecolor='cyan', alpha=0.4)
        if not X_est_KF is None:
            ax.plot(X_est_KF[T_start:T_end,idim], 'gv-',label='$\\hat{\mathbf{x}}_{KF}$',lw=lw, ms=1)
            ax.fill_between(np.arange(X_est_KF[T_start:T_end,idim].shape[0]), 
                            X_est_KF[T_start:T_end,idim] - X_est_KF_std[T_start:T_end,idim], 
                            X_est_KF[T_start:T_end,idim] + X_est_KF_std[T_start:T_end,idim],
                            facecolor='green', alpha=0.4)
        if not X_est_EKF is None:
            ax.plot(X_est_EKF[T_start:T_end,idim], 'b.-',label='$\\hat{\mathbf{x}}_{EKF}$',lw=lw)
            ax.fill_between(np.arange(X_est_EKF[T_start:T_end,idim].shape[0]), 
                            X_est_EKF[T_start:T_end,idim] - X_est_EKF_std[T_start:T_end,idim], 
                            X_est_EKF[T_start:T_end,idim] + X_est_EKF_std[T_start:T_end,idim],
                            facecolor='orange', alpha=0.4)
        
        ax.plot(X[T_start:T_end,idim],'k-',label='$\\mathbf{x}^{true}$',lw=lw)

        ax.set_ylabel('$x_{}$'.format(idim+1))
        ax.set_xlabel('$t$')
        #plt.legend()
        #handles, labels = ax.get_legend_handles_labels()
        #order=None
        #if order is None:
        #    order=range(len(handles))
        #ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],ncol=5,loc=(-0.02,1.01),fontsize=14)
        plt.legend()
        plt.tight_layout()
        
    elif X.shape[-1] > 2:
        T_start= 650 #33
        T_end= 1000 #165
        idim=1
        lw=1.3
        #plt.rcParams['font.size'] = 16
        plt.rcParams['font.family']='serif'
        #fig, ax = plt.subplots(figsize=(12,7))
        fig, ax = plt.subplots(figsize=(20,7))
        #plt.subplot(311)
        if not X_est_UKF is None:
            ax.plot(X_est_UKF[T_start:T_end,idim], 'x-',ms=5,color="orange",label='$\\hat{\mathbf{x}}_{UKF}$',lw=lw)
            ax.fill_between(np.arange(X_est_UKF[T_start:T_end,idim].shape[0]), 
                            X_est_UKF[T_start:T_end,idim] - 3*X_est_UKF_std[T_start:T_end,idim], 
                            X_est_UKF[T_start:T_end,idim] + 3*X_est_UKF_std[T_start:T_end,idim],
                            facecolor='orange', alpha=0.4)
        if not X_est_DANSE is None:
            ax.plot(X_est_DANSE[T_start:T_end,idim], 'rs-',label='$\\hat{\mathbf{x}}_{DANSE}$',lw=lw, ms=2)
            ax.fill_between(np.arange(X_est_DANSE[T_start:T_end,idim].shape[0]), 
                            X_est_DANSE[T_start:T_end,idim] - 3*X_est_DANSE_std[T_start:T_end,idim], 
                            X_est_DANSE[T_start:T_end,idim] + 3*X_est_DANSE_std[T_start:T_end,idim],
                            facecolor='red', alpha=0.4)
        if not X_est_DANSE_sup is None:
            ax.plot(X_est_DANSE_sup[T_start:T_end,idim], 'm^-',label='$\\hat{\mathbf{x}}_{DANSE-Supervised}$',lw=lw, ms=3)
            ax.fill_between(np.arange(X_est_DANSE_sup[T_start:T_end,idim].shape[0]), 
                            X_est_DANSE_sup[T_start:T_end,idim] - 3*X_est_DANSE_sup_std[T_start:T_end,idim], 
                            X_est_DANSE_sup[T_start:T_end,idim] + 3*X_est_DANSE_sup_std[T_start:T_end,idim],
                            facecolor='magenta', alpha=0.4)
        if not X_est_KNET is None:
            plt.plot(X_est_KNET[T_start:T_end,idim], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet} $', lw=lw)
            ax.fill_between(np.arange(X_est_KNET[T_start:T_end,idim].shape[0]), 
                            X_est_KNET[T_start:T_end,idim] - 3*X_est_KNET_std[T_start:T_end,idim], 
                            X_est_KNET[T_start:T_end,idim] + 3*X_est_KNET_std[T_start:T_end,idim],
                            facecolor='cyan', alpha=0.4)
        if not X_est_KF is None:
            ax.plot(X_est_KF[T_start:T_end,idim], 'gv-',label='$\\hat{\mathbf{x}}_{KF}$',lw=lw, ms=1)
            ax.fill_between(np.arange(X_est_KF[T_start:T_end,idim].shape[0]), 
                            X_est_KF[T_start:T_end,idim] - 3*X_est_KF_std[T_start:T_end,idim], 
                            X_est_KF[T_start:T_end,idim] + 3*X_est_KF_std[T_start:T_end,idim],
                            facecolor='green', alpha=0.4)
        if not X_est_EKF is None:
            ax.plot(X_est_EKF[T_start:T_end,idim], 'b.-',label='$\\hat{\mathbf{x}}_{EKF}$',lw=lw)
            ax.fill_between(np.arange(X_est_EKF[T_start:T_end,idim].shape[0]), 
                            X_est_EKF[T_start:T_end,idim] - 3*X_est_EKF_std[T_start:T_end,idim], 
                            X_est_EKF[T_start:T_end,idim] + 3*X_est_EKF_std[T_start:T_end,idim],
                            facecolor='orange', alpha=0.4)
        
        ax.plot(X[T_start:T_end,idim],'k-',label='$\\mathbf{x}^{true}$',lw=lw)

        ax.set_ylabel('$x_{}$'.format(idim+1))
        ax.set_xlabel('$t$')
        #plt.legend()
        #handles, labels = ax.get_legend_handles_labels()
        #order=None
        #if order is None:
        #    order=range(len(handles))
        #ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],ncol=5,loc=(-0.02,1.01),fontsize=14)
        plt.legend()
        plt.tight_layout()

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig_name)
        tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")
    #plt.show()
    return None

def plot_3d_state_trajectory(X, legend='$\\mathbf{x}^{true}$', m='k-', savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    #print(X.shape)
    
    if X.shape[-1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X[:,0], X[:,1], m, label=legend)
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        plt.legend()
        
    elif X.shape[-1] > 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:,0], X[:,1], X[:,2], m)#, label=legend)
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        ax.set_zlabel('$X_3$')
        #plt.legend()
        plt.tight_layout()

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig_name)
        #tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")
    #plt.show()
    return None

def plot_3d_measurment_trajectory(Y, legend='$\\mathbf{y}$', m='k-', savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    #print(X.shape)
    
    if Y.shape[-1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(Y[:,0], Y[:,1], m, label=legend)
        ax.set_xlabel('$Y_1$')
        ax.set_ylabel('$Y_2$')
        plt.legend()
        
    elif Y.shape[-1] > 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(Y[:,0], Y[:,1], Y[:,2], m)#, label=legend)
        ax.set_xlabel('$Y_1$')
        ax.set_ylabel('$Y_2$')
        ax.set_zlabel('$Y_3$')
        #plt.legend()
        plt.tight_layout()

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig_name)
        #tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")
    #plt.show()
    return None

def plot_measurement_data(Y, savefig=False, savefig_name=None):
    
    # Plot the measurement data
    fig = plt.figure()

    if Y.shape[-1] == 2:

        ax = fig.add_subplot(111)
        ax.plot(Y[:,0], Y[:,1], '--', label='$\\mathbf{y}^{measured}$')
        ax.set_xlabel('$Y_1$')
        ax.set_ylabel('$Y_2$')
        plt.legend()
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig_name)
            tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")

    elif Y.shape[-1] > 2:

        ax = fig.add_subplot(111, projection='3d')
        ax.plot(Y[:,0], Y[:,1], Y[:, 2], '--', label='$\\mathbf{y}^{measured}$')
        ax.set_xlabel('$Y_1$')
        ax.set_ylabel('$Y_2$')
        ax.set_zlabel('$Y_3$')
        plt.legend()
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig_name)
            tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")

    #plt.show()
    return None

def plot_state_trajectory_axes(X, X_est_KF=None, X_est_EKF=None, X_est_UKF=None, X_est_KNET=None, X_est_DANSE=None, X_est_DANSE_sup=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    #print(X.shape)
    Tx, _ = X.shape
    T_end = 200

    if X.shape[-1] == 2:
        fig = plt.figure(figsize=(20,10))
        plt.subplot(311)
        plt.plot(X[:T_end,0],'--',label='$\\mathbf{x}^{true} (x-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:T_end,0], 'g:',label='$\\hat{\mathbf{x}}_{KF} (x-component) $')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:T_end,0], 'r--',label='$\\hat{\mathbf{x}}_{DANSE} (x-component) $')
        if not X_est_DANSE_sup is None:
            plt.plot(X_est_DANSE_sup[:T_end,0], 'm--',label='$\\hat{\mathbf{x}}_{DANSE-Supervised} (x-component) $')
        if not X_est_KNET is None:
            plt.plot(X_est_KNET[:T_end,0], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet} (x-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:T_end,0], 'b.-',label='$\\hat{\mathbf{x}}_{EKF} (x-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:T_end,0], '-x',ms=4,color="orange",label='$\\hat{\mathbf{x}}_{UKF} (x-component) $')
        plt.ylabel('$X_1$')
        plt.xlabel('$t$')
        plt.legend()

        plt.subplot(312)
        plt.plot(X[:T_end,1], '--',label='$\\mathbf{x}^{true} (y-component)$')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:T_end,1], 'r--',label='$\\hat{\mathbf{x}}_{DANSE} (y-component) $')
        if not X_est_DANSE_sup is None:
            plt.plot(X_est_DANSE_sup[:T_end,1], 'm--',label='$\\hat{\mathbf{x}}_{DANSE-Supervised} (y-component) $')
        if not X_est_KNET is None:
            plt.plot(X_est_KNET[:T_end,1], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet} (y-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:T_end,1], 'g:',label='$\\hat{\mathbf{x}}_{KF} (y-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:T_end,1], 'b.-',label='$\\hat{\mathbf{x}}_{EKF} (y-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:T_end,1], 'x-',ms=4,color="orange",label='$\\hat{\mathbf{x}}_{UKF} (y-component) $')
        plt.ylabel('$X_2$')
        plt.xlabel('$t$')
        plt.legend()
        
    elif X.shape[-1] > 2:
        T_start = 650 #33
        T_end = 1000 #165
        idim=1
        lw=1.3
        plt.rcParams['font.size'] = 16
        #plt.rcParams['font.family']='serif'
        fig, ax = plt.subplots(figsize=(12,7))
        #plt.subplot(311)
        if not X_est_UKF is None:
            ax.plot(X_est_UKF[T_start:T_end,idim], 'x-',ms=5,color="orange",label='$\\hat{\mathbf{x}}_{UKF}$',lw=lw)
        if not X_est_DANSE is None:
            ax.plot(X_est_DANSE[T_start:T_end,idim], 'rs-',label='$\\hat{\mathbf{x}}_{DANSE}$',lw=lw, ms=4)
        if not X_est_DANSE_sup is None:
            ax.plot(X_est_DANSE_sup[T_start:T_end,idim], 'm^-',label='$\\hat{\mathbf{x}}_{DANSE-Supervised}$',lw=lw, ms=3)
        if not X_est_KNET is None:
            plt.plot(X_est_KNET[T_start:T_end,idim], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet} $', lw=lw)
        if not X_est_KF is None:
            ax.plot(X_est_KF[T_start:T_end,idim], 'g:',label='$\\hat{\mathbf{x}}_{KF}$',lw=lw)
        if not X_est_EKF is None:
            ax.plot(X_est_EKF[T_start:T_end,idim], 'b.-',label='$\\hat{\mathbf{x}}_{EKF}$',lw=lw)
        ax.plot(X[T_start:T_end,idim],'k-',label='$\\mathbf{x}^{true}$',lw=lw)

        ax.set_ylabel('$x_{}$'.format(idim+1))
        ax.set_xlabel('$t$')
        #plt.legend()
        #handles, labels = ax.get_legend_handles_labels()
        #order=None
        #if order is None:
        #    order=range(len(handles))
        #ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],ncol=5,loc=(-0.02,1.01),fontsize=14)
        plt.legend()
        plt.tight_layout()
        
        '''
        plt.subplot(312)
        plt.plot(X[:T_end,1], '--',label='$\\mathbf{x}^{true} (y-component)$')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:T_end,1], '--',label='$\\hat{\mathbf{x}}_{DANSE} (y-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:T_end,1], ':',label='$\\hat{\mathbf{x}}_{KF} (y-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:T_end,1], ':',label='$\\hat{\mathbf{x}}_{EKF} (y-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:T_end,1], ':',label='$\\hat{\mathbf{x}}_{UKF} (y-component) $')
        plt.ylabel('$X_2$')
        plt.xlabel('$t$')
        plt.legend()
    
        plt.subplot(313)
        plt.plot(X[:T_end,2],'--',label='$\\mathbf{x}^{true} (z-component)$')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:T_end,2], '--',label='$\\hat{\mathbf{x}}_{DANSE} (z-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:T_end,2], ':',label='$\\hat{\mathbf{x}}_{KF} (z-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:T_end,2], ':',label='$\\hat{\mathbf{x}}_{EKF} (z-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:T_end,2], ':',label='$\\hat{\mathbf{x}}_{UKF} (z-component) $')
        plt.ylabel('$X_3$')
        plt.xlabel('$t$')
        plt.legend()
        '''
    plt.tight_layout()
    if savefig:
        fig.savefig(savefig_name,dpi=300,bbox_inches="tight")
        tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")
    #plt.show()
    return None

def plot_state_trajectory_axes_all(X, X_est_KF=None, X_est_EKF=None, X_est_UKF=None, X_est_KNET=None, X_est_DANSE=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    #print(X.shape)
    Tx, _ = X.shape
    T_end = 200

    if X.shape[-1] == 2:
        fig = plt.figure(figsize=(20,10))
        plt.subplot(311)
        plt.plot(X[:T_end,0],'--',label='$\\mathbf{x}^{true} (x-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:T_end,0], 'g:',label='$\\hat{\mathbf{x}}_{KF} (x-component) $')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:T_end,0], 'r--',label='$\\hat{\mathbf{x}}_{DANSE} (x-component) $')
        if not X_est_KNET is None:
            plt.plot(X_est_KNET[:T_end,0], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet} (x-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:T_end,0], 'b.-',label='$\\hat{\mathbf{x}}_{EKF} (x-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:T_end,0], '-x',ms=4,color="orange",label='$\\hat{\mathbf{x}}_{UKF} (x-component) $')
        plt.ylabel('$X_1$')
        plt.xlabel('$t$')
        plt.legend()

        plt.subplot(312)
        plt.plot(X[:T_end,1], '--',label='$\\mathbf{x}^{true} (y-component)$')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:T_end,1], 'r--',label='$\\hat{\mathbf{x}}_{DANSE} (y-component) $')
        if not X_est_KNET is None:
            plt.plot(X_est_KNET[:T_end,1], 'c--.',label='$\\hat{\mathbf{x}}_{KalmanNet} (y-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:T_end,1], 'g:',label='$\\hat{\mathbf{x}}_{KF} (y-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:T_end,1], 'b.-',label='$\\hat{\mathbf{x}}_{EKF} (y-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:T_end,1], 'x-',ms=4,color="orange",label='$\\hat{\mathbf{x}}_{UKF} (y-component) $')
        plt.ylabel('$X_2$')
        plt.xlabel('$t$')
        plt.legend()
        
    elif X.shape[-1] > 2:
        
        fig = plt.figure(figsize=(20,10))
        plt.subplot(311)
        plt.plot(X[:T_end,0], '--',label='$\\mathbf{x}^{true} (x-component)$')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:T_end,0], '--',label='$\\hat{\mathbf{x}}_{DANSE} (x-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:T_end,0], ':',label='$\\hat{\mathbf{x}}_{KF} (x-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:T_end,0], ':',label='$\\hat{\mathbf{x}}_{EKF} (x-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:T_end,0], ':',label='$\\hat{\mathbf{x}}_{UKF} (x-component) $')
        plt.ylabel('$X_1$')
        plt.xlabel('$t$')
        plt.legend()

        plt.subplot(312)
        plt.plot(X[:T_end,1], '--',label='$\\mathbf{x}^{true} (y-component)$')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:T_end,1], '--',label='$\\hat{\mathbf{x}}_{DANSE} (y-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:T_end,1], ':',label='$\\hat{\mathbf{x}}_{KF} (y-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:T_end,1], ':',label='$\\hat{\mathbf{x}}_{EKF} (y-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:T_end,1], ':',label='$\\hat{\mathbf{x}}_{UKF} (y-component) $')
        plt.ylabel('$X_2$')
        plt.xlabel('$t$')
        plt.legend()
    
        plt.subplot(313)
        plt.plot(X[:T_end,2],'--',label='$\\mathbf{x}^{true} (z-component)$')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:T_end,2], '--',label='$\\hat{\mathbf{x}}_{DANSE} (z-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:T_end,2], ':',label='$\\hat{\mathbf{x}}_{KF} (z-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:T_end,2], ':',label='$\\hat{\mathbf{x}}_{EKF} (z-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:T_end,2], ':',label='$\\hat{\mathbf{x}}_{UKF} (z-component) $')
        plt.ylabel('$X_3$')
        plt.xlabel('$t$')
        plt.legend()
        
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig_name,dpi=300,bbox_inches="tight")
        #tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")
    #plt.show()
    return None

def plot_state_trajectory_pred(X, X_est=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    print(X.shape)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'
    if X.shape[-1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X[:,0], X[:,1],'--',label='$\\mathbf{x}_{DANSE}$')
        if not X_est is None:
            ax.plot(X_est[:,0], X_est[:,1],'--',label='$\\hat{\mathbf{x}}$')
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        plt.legend()
        
    elif X.shape[-1] > 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:,0], X[:,1], X[:,2], '--',label='$\\mathbf{x}_{DANSE}$')
        if not X_est is None:
            ax.plot(X_est[:,0], X_est[:,1], X_est[:,2], '--',label='$\\hat{\mathbf{x}}$')
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        ax.set_zlabel('$X_3$')
        plt.legend()
    
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig_name)
    plt.show()
    return None

def plot_measurement_data_axes(Y, Y_est=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    fig = plt.figure(figsize=(20,10))

    if Y.shape[-1] == 2:
        plt.subplot(311)
        plt.plot(Y[:,0],'--',label='$\\mathbf{Y}^{true} (x-component) $')
        if not Y_est is None:
            plt.plot(Y_est[:,0], '--',label='$\\hat{\mathbf{Y}} (x-component) $')
        plt.ylabel('$Y_1$')
        plt.xlabel('$t$')
        plt.legend()
        
        plt.subplot(312)
        plt.plot(Y[:,1], '--',label='$\\mathbf{Y}^{true} (y-component)$')
        if not Y_est is None:
            plt.plot(Y_est[:,1], '--',label='$\\hat{\mathbf{Y}} (y-component)$')
        plt.ylabel('$Y_2$')
        plt.xlabel('$t$')
        plt.legend()

    elif Y.shape[-1] > 2:
        plt.subplot(311)
        plt.plot(Y[:,0],'--',label='$\\mathbf{Y}^{true} (x-component) $')
        if not Y_est is None:
            plt.plot(Y_est[:,0], '--',label='$\\hat{\mathbf{Y}} (x-component) $')
        plt.ylabel('$Y_1$')
        plt.xlabel('$t$')
        plt.legend()
        
        plt.subplot(312)
        plt.plot(Y[:,1], '--',label='$\\mathbf{Y}^{true} (y-component)$')
        if not Y_est is None:
            plt.plot(Y_est[:,1], '--',label='$\\hat{\mathbf{Y}} (y-component)$')
        plt.ylabel('$Y_2$')
        plt.xlabel('$t$')
        plt.legend()

        plt.subplot(313)
        plt.plot(Y[:,2],'--',label='$\\mathbf{Y}^{true} (z-component)$')
        if not Y_est is None:
            plt.plot(Y_est[:,2],'--',label='$\\hat{\mathbf{Y}} (z-component)$')
        plt.ylabel('$Y_3$')
        plt.xlabel('$t$')
        plt.legend()
    
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig_name)
        tikzplotlib.save(os.path.splitext(savefig_name)[0] + ".tex")
    #plt.show()
    return None
