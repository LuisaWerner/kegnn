import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
import json

api = wandb.Api()


# only for citeseer

#three layers

# clause weight 0.25, 3 layers
dainty24=api.run("luisawerner/ijcai23_citeseer_kenngcn_2/yqyplmqf").summary['logged_clause_weights']

# clause weight 0.5, 3 layers
skilled=api.run("luisawerner/ijcai23_citeseer_kenngcn_2/8k9m5a8i").summary['logged_clause_weights']

# clause weight 0.001, 3 layers
smooth87=api.run("luisawerner/ijcai23_citeseer_kenngcn_2/my2jz6iq").summary['logged_clause_weights']

# clause weight 0.1, 3 layers
jolly=api.run("luisawerner/ijcai23_citeseer_kenngcn_2/j1wph4m6").summary['logged_clause_weights']

# clause weight -0.5, 3 layers
deep = api.run("luisawerner/ijcai23_citeseer_kenngcn_2/vdjwae4h").summary['logged_clause_weights']




skilled_weights = eval(skilled)
dainty = eval (dainty24)


plt.figure()
color = plt.cm.rainbow(np.linspace(0, 1, 6))
patches = []
for k, j in enumerate(['0', '1', '2', '3', '4', '5']):
    patches.append(mpatches.Patch(color=color[k], label='Class_'+j))
    lista = [0.5] + skilled_weights[0][0][j]
    plt.plot(lista, color=color[k])
    # for i, v in enumerate(skilled_weights[0][0][j]):
    #     plt.scatter(i, v, color=color[k])


plt.legend(handles=patches)
plt.ylim(0.00, 0.6)
plt.savefig('test_clause_weights, layer 0 ')


patches = []
for k, j in enumerate(['0', '1', '2', '3', '4', '5']):
    patches.append(mpatches.Patch(color=color[k], label='Class_'+j))
    lista = [0.5] + skilled_weights[0][1][j]
    plt.plot(lista, color=color[k])
    # for i, v in enumerate(skilled_weights[0][0][j]):
    #     plt.scatter(i, v, color=color[k])


plt.legend(handles=patches)
plt.ylim(0.00, 0.6)
plt.savefig('test_clause_weights_layer2')

patches = []
for k, j in enumerate(['0', '1', '2', '3', '4', '5']):
    patches.append(mpatches.Patch(color=color[k], label='Class_'+j))
    lista = [0.5] + skilled_weights[0][2][j]
    plt.plot(lista, color=color[k])
    # for i, v in enumerate(skilled_weights[0][0][j]):
    #     plt.scatter(i, v, color=color[k])


plt.legend(handles=patches)
plt.ylim(0.00, 0.6)
plt.savefig('test_clause_weights_layer3')


print('test')




plt.figure()
color = plt.cm.rainbow(np.linspace(0, 1, 6))
patches = []
for k, j in enumerate(['0', '1', '2', '3', '4', '5']):
    patches.append(mpatches.Patch(color=color[k], label='Class_'+j))
    lista = [0.25] + dainty[0][0][j]
    plt.plot(lista, color=color[k])
    # for i, v in enumerate(skilled_weights[0][0][j]):
    #     plt.scatter(i, v, color=color[k])


plt.legend(handles=patches)
#plt.ylim(0.00, 0.6)
plt.savefig('test_clause_weights_layer0_025 ')


patches = []
for k, j in enumerate(['0', '1', '2', '3', '4', '5']):
    patches.append(mpatches.Patch(color=color[k], label='Class_'+j))
    lista = [0.25] + dainty[0][1][j]
    plt.plot(lista, color=color[k])
    # for i, v in enumerate(skilled_weights[0][0][j]):
    #     plt.scatter(i, v, color=color[k])


plt.legend(handles=patches)
#plt.ylim(0.00, 0.6)
plt.savefig('test_clause_weights_layer2_025')

patches = []
for k, j in enumerate(['0', '1', '2', '3', '4', '5']):
    patches.append(mpatches.Patch(color=color[k], label='Class_'+j))
    lista = [0.25] + dainty[0][2][j]
    plt.plot(lista, color=color[k])
    # for i, v in enumerate(skilled_weights[0][0][j]):
    #     plt.scatter(i, v, color=color[k])


plt.legend(handles=patches)
#plt.ylim(0.00, 0.6)
plt.savefig('test_clause_weights_layer3_025')