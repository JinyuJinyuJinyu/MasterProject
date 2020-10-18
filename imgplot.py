import matplotlib.pyplot as plt
import numpy as np
import json

#airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

f = open('test.json','r',encoding='utf-8')
data = json.load(f)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

confusion = [[
                763.0,
                11.0,
                68.0,
                19.0,
                15.0,
                4.0,
                15.0,
                10.0,
                84.0,
                11.0
            ],
            [
                10.0,
                874.0,
                2.0,
                6.0,
                2.0,
                4.0,
                12.0,
                3.0,
                26.0,
                61.0
            ],
            [
                45.0,
                3.0,
                661.0,
                64.0,
                77.0,
                49.0,
                60.0,
                13.0,
                16.0,
                12.0
            ],
            [
                16.0,
                6.0,
                66.0,
                592.0,
                44.0,
                168.0,
                68.0,
                17.0,
                14.0,
                9.0
            ],
            [
                14.0,
                4.0,
                56.0,
                60.0,
                734.0,
                47.0,
                43.0,
                31.0,
                9.0,
                2.0
            ],
            [
                13.0,
                1.0,
                48.0,
                174.0,
                33.0,
                673.0,
                24.0,
                25.0,
                7.0,
                2.0
            ],
            [
                7.0,
                0.0,
                27.0,
                55.0,
                30.0,
                22.0,
                846.0,
                5.0,
                6.0,
                2.0
            ],
            [
                19.0,
                2.0,
                35.0,
                46.0,
                48.0,
                54.0,
                6.0,
                777.0,
                4.0,
                9.0
            ],
            [
                25.0,
                14.0,
                6.0,
                10.0,
                2.0,
                5.0,
                11.0,
                2.0,
                917.0,
                8.0
            ],
            [
                36.0,
                44.0,
                3.0,
                14.0,
                5.0,
                10.0,
                7.0,
                5.0,
                32.0,
                844.0
            ]]

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

    plt.rcParams["font.size"] = "1"
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

accu = []
for epoch in data:
    if 'test acc' in epoch:
        accu.append(epoch["test acc"])


plt.plot(accu)
plt.show()

def vis_accu(accu):
    plt.plot(accu)
    plt.show()


# vis_confusion(confusion)