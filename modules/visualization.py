from os import path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

class Plotter():

    def __init__(self, save_dir='.'):

        self.save_dir = save_dir

    def plot_pr(self, precision, recall, thres_list, name=None,
                filename_suffix=None):

        fig = plt.figure('pr', figsize=(5,5), dpi=300, constrained_layout=True)
        ax = fig.gca()

        # f1 contour

        levels = 10

        spacing = np.linspace(0, 1, 1000)
        x, y = np.meshgrid(spacing, spacing)

        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 / (1/x + 1/y)

        locx = np.linspace(0, 1, levels, endpoint=False)[1:]

        cs = ax.contour(x, y, f1, levels=levels, linewidths=1, colors='k',
                        alpha=0.3)
        ax.clabel(cs, inline=True, fmt='F1=%.1f',
                  manual=np.tile(locx,(2,1)).T)

        with np.errstate(divide='ignore', invalid='ignore'):
            aupr = np.trapz(np.flip(precision), x=np.flip(recall))
            f1 = 2*recall*precision / (recall+precision)
        f1_max_idx = np.nanargmax(f1)
        f1_max = f1[f1_max_idx]

        ax.plot(recall, precision, lw=1, color='C0', label=name)

        ax.scatter(recall[f1_max_idx], precision[f1_max_idx],
                   label=f'{thres_list[f1_max_idx]:.4f} (f1$_{{max}}$)',
                   marker='o', edgecolors='C1',
                   facecolors='none', linewidths=0.5)
        if thres_list.size%2 == 1:
            def_thres_idx = (thres_list.size-1)//2
            ax.scatter(recall[def_thres_idx], precision[def_thres_idx],
                       label=f'{0.5:.4f}',
                       marker='x', c='C1',
                       linewidths=0.5)
        plt.legend(title='threshold')

        plt.xlabel('recall')
        plt.ylabel('precision')

        # plt.title(f'AUPR: {aupr}, f1: {f1_max}')
        plt.title(f'f1$_{{max}}$: {f1_max:.6f}')

        # plt.show()
        filename = (f'pr_curve-{filename_suffix}.png'
                        if filename_suffix else 'pr_curve.png')
        plt.savefig(path.join(self.save_dir, filename))

        plt.close()

    def plot_loss_acc_f1_hist(self, loss_acc_f1_hist, filename_suffix=None):

        n_epochs = loss_acc_f1_hist.shape[0]
        epoch = np.arange(1, n_epochs+1)

        # plot loss
        fig = plt.figure('loss', figsize=(6,4), dpi=300,
                         constrained_layout=True)
        plt.plot(epoch, loss_acc_f1_hist[:,0], label='train')
        plt.plot(epoch, loss_acc_f1_hist[:,1], label='valid')
        plt.xlabel('epoch number')
        plt.ylabel('loss')
        plt.xlim(0, n_epochs)
        plt.legend()
        plt.grid()
        filename = 'training_hist-loss.png'
        plt.savefig(path.join(self.save_dir, filename))
        plt.clf()

        # plot accuracy
        fig, ax1 = plt.subplots(figsize=(6,4), dpi=300,
                                constrained_layout=True, num='acc')
        ax2 = ax1.twinx()

        # plot accuracy
        ax1.set_ylabel('accuracy')
        ax1.plot(epoch, loss_acc_f1_hist[:,2], c='C0', label='train acc')
        ax1.plot(epoch, loss_acc_f1_hist[:,3], c='C1', label='valid acc')
        ax1.set_xlabel('epoch number')
        ax1.set_xlim(0, n_epochs)
        ax1.set_ylim(0,1)
        ax1.grid()

        # plot f1
        ax2.set_ylabel('f1 (not f1$_{max}$)')
        ax2.plot(epoch, loss_acc_f1_hist[:,4], '--', c='C0')
        ax2.plot(epoch, loss_acc_f1_hist[:,5], '--', c='C1')
        ax2.set_ylim(0,1)
        ax2.grid(False)

        # set up legends
        train = mlines.Line2D([], [], color='C0', ls='-', label='train')
        valid = mlines.Line2D([], [], color='C1', ls='-', label='valid')
        acc   = mlines.Line2D([], [],  color='black', ls='-', label='acc')
        f1    = mlines.Line2D([], [],  color='black', ls='--', label='f1')
        plt.legend(handles=[train, valid, acc, f1], ncol=2)

        filename = 'training_hist-metrics.png'
        plt.savefig(path.join(self.save_dir, filename))
        plt.close()

    def plot_f1_max_hist(self, f1_max_hist, filename_suffix=None):

        n_epochs = f1_max_hist.shape[0]
        epoch = np.arange(1, n_epochs+1)

        # plot loss
        fig, ax1 = plt.subplots(figsize=(6,4), dpi=300,
                                constrained_layout=True, num='f1')
        ax2 = ax1.twinx()

        ax1.set_ylabel('f1$_{max}$ / precision / recall / accuracy')
        ax1.plot(epoch, f1_max_hist[:,1],
                 label='f1$_{max}$')
        ax1.plot(epoch, f1_max_hist[:,2],
                 label='ppv @ f1$_{max}$', alpha=0.9)
        ax1.plot(epoch, f1_max_hist[:,3],
                 label='tpr @ f1$_{max}$', alpha=0.9)
        ax1.plot(epoch, f1_max_hist[:,4],
                 label='acc @ f1$_{max}$', alpha=0.9)
        ax1.legend()
        ax1.set_xlim(0, n_epochs)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('epoch number')
        ax1.grid()

        ax2.plot(epoch, f1_max_hist[:,0], '--', c='C5', alpha=0.9)
        ax2.set_ylabel('threshold @ f1$_{max}$', color='C5')
        ax2.set_ylim(0, 1)
        ax2.grid(False)

        plt.grid()
        filename = 'training_hist-f1_max.png'
        plt.savefig(path.join(self.save_dir, filename))
        plt.close()
