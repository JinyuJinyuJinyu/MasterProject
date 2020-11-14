import matplotlib.pyplot as plt
import numpy as np
import json

#airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck




classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def vis_confusion(confusion):
    confusion = np.matrix(confusion)
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    num_marks = np.arange(len(classes))
    plt.xticks(num_marks, classes, rotation=45)
    plt.yticks(num_marks, classes)

    for i in range(10):
        for j in range(10):
            plt.text(j, i, confusion[i, j],
                     horizontalalignment="center",
                     color="black")
    plt.title('trained with Adam, epoch: 4, accu: 45.41')
    plt.rcParams["font.size"] = "1"
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def vis_accu():

    files = ['adam_VGG16_tf.json','SGD_VGG16_tf.json','adam_VGG16_tr.json','SGD_VGG16_tr.json']
    for jsf in files:
        f = open(jsf,'r',encoding='utf-8')
        data = json.load(f)
        accu = []
        for epoch in data:
            if 'accu' in epoch:
                accu.append(epoch["accu"])
        print(accu)
        plt.plot(np.linspace(1, 201, num=200), accu, label=jsf.split('.')[0],alpha=1)
        print(jsf.split('.')[0])
    plt.legend(loc="upper right")
    plt.ylim(ymin=0)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    # plt.savefig()
    plt.show()

def time_diff():
    labels = ['adam_TF', 'sgd_TF','adam_PyTorch', 'sgd_PyTorch']
    times = [120.5,117.9,347.19,282.59]
    init_times = [137.67,133.08,371.44,301.96]

    x = np.arange(len(labels))

    width = 0.8
    fig, ax = plt.subplots()
    plt.title('init epoch time difference')
    barlist = ax.bar(labels, init_times, width)
    barlist[0].set_color('#1f77b4')
    barlist[1].set_color('#ff7f0e')
    barlist[2].set_color('#FFBB33')
    barlist[3].set_color('#33FF41')
    ax.set_ylabel('time')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('framework')

    plt.show()
