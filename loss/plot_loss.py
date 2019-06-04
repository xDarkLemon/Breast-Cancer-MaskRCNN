import re
import matplotlib.pyplot as plt

# with open('loss_new.txt', 'w') as f0:
#     with open('loss.txt', 'r') as f1:
#         for line in f1.readlines():
#             if line.startswith('985/985'):
#                 f0.write(line)

def showPlot(plot_epoches, plot_losses, plot_val_losses, i, dataset=None, learning_rate=None, batch_size=None):
    plt.subplot(2,3,i+1)
    plt.plot(plot_epoches, plot_losses, color='red', label='train loss')
    plt.plot(plot_epoches, plot_val_losses, color='blue', label='validate loss')
    plt.ylim(0, )
    # title = str(dataset).split('_')[1]+'_lr='+str(learning_rate)+'_batchsize='+str(batch_size)+'_epoch='+str(plot_epoches[-1])
    name = ['loss', 'rpn_class_loss', 'rpn_bbox_loss', 'mrcnn_class_loss', 'mrcnn_bbox_loss', 'mrcnn_mask_loss', \
            'val_loss', 'val_rpn_class_loss', 'val_rpn_bbox_loss', 'val_mrcnn_class_loss', 'val_mrcnn_bbox_loss', 'val_mrcnn_mask_loss'\
            ]
    title = name[i]
    plt.title(title)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(title + '.jpg')
    # plt.show()
    return

def plot_loss(i):
    with open('loss.txt', 'r') as f2:
        epoch = 0
        epoches = []
        losses = []
        val_losses = []
        for line in f2.readlines():
            epoch += 1
            epoches.append(epoch)
            # loss, rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, \
            # val_loss, val_rpn_class_loss, val_rpn_bbox_loss, val_mrcnn_class_loss, val_mrcnn_bbox_loss, val_mrcnn_mask_loss\
            all_loss = re.findall('\d+\.\d+', line)
            all_loss = [float(loss) for loss in all_loss]
            losses.append(all_loss[i])
            val_losses.append(all_loss[i+6])
        print(losses)
        print(val_losses)
        print(epoches)

    showPlot(epoches, losses, val_losses, i)
    
for i in range(6):
    plot_loss(i)
plt.show()