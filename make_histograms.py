import wandb
import numpy as np
import matplotlib.pyplot as plt

api = wandb.Api()

#cora
cora_mlp=api.run("luisawerner/ijcai23/2343ecup").summary['test_accuracies']
cora_kenn_mlp=api.run("luisawerner/ijcai23/ji4ffi8b").summary['test_accuracies']
cora_gcn=api.run("luisawerner/ijcai23/6shv0saa").summary['test_accuracies']
cora_kenn_gcn=api.run("luisawerner/ijcai23/3eye9d2f").summary['test_accuracies']

#citeseer
citeseer_mlp=api.run("luisawerner/ijcai23/dg8asive").summary['test_accuracies']
citeseer_kenn_mlp=api.run("luisawerner/ijcai23/3fh9dm9f").summary['test_accuracies']
citeseer_gcn=api.run("luisawerner/ijcai23/2t1rhmgg").summary['test_accuracies']
citeseer_kenn_gcn=api.run("luisawerner/ijcai23/26e1ecjo").summary['test_accuracies']

#pubmed
pubmed_mlp=api.run("luisawerner/ijcai23/11tzslov").summary['test_accuracies']
pubmed_kenn_mlp=api.run("luisawerner/ijcai23/1gxdt96u").summary['test_accuracies']
pubmed_gcn=api.run("luisawerner/ijcai23/yyqqkqgt").summary['test_accuracies']
pubmed_kenn_gcn=api.run("luisawerner/ijcai23/r3n99eqd").summary['test_accuracies']

#flickr
flickr_mlp=api.run("luisawerner/ijcai23/27trpunh").summary['test_accuracies']
flickr_kenn_mlp=api.run("luisawerner/ijcai23/3myuzhfp").summary['test_accuracies']
flickr_gcn=api.run("luisawerner/ijcai23/30cv2k54").summary['test_accuracies']
flickr_kenn_gcn=api.run("luisawerner/ijcai23/1w4ahs05").summary['test_accuracies']


# Build the plot for cora
xpos = np.arange(4)
means = [np.mean(cora_mlp), np.mean(cora_kenn_mlp), np.mean(cora_gcn), np.mean(cora_kenn_gcn)]
errors = [np.std(cora_mlp), np.std(cora_kenn_mlp), np.std(cora_gcn), np.std(cora_kenn_gcn)]
fig, ax = plt.subplots()
ax.bar(xpos, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('test accuracy')
ax.set_xticks(xpos)
ax.set_xticklabels(['MLP', 'KENN_MLP', 'GCN', 'KENN_GCN'])
ax.set_title('Cora Test accuracy plot ')
ax.yaxis.grid(True)

# Save the figure and show
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig('cora_error_bars.png')
plt.show()

#
means = [np.mean(citeseer_mlp), np.mean(citeseer_kenn_mlp), np.mean(citeseer_gcn), np.mean(citeseer_kenn_gcn)]
errors = [np.std(citeseer_mlp), np.std(citeseer_kenn_mlp), np.std(citeseer_gcn), np.std(citeseer_kenn_gcn)]
fig, ax = plt.subplots()
ax.bar(xpos, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('test accuracy')
ax.set_xticks(xpos)
ax.set_xticklabels(['MLP', 'KENN_MLP', 'GCN', 'KENN_GCN'])
ax.set_title('Citeseer Test accuracy plot ')
ax.yaxis.grid(True)

# Save the figure and show
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig('citeseer_error_bars.png')
plt.show()

#pubmed
means = [np.mean(pubmed_mlp), np.mean(pubmed_kenn_mlp), np.mean(pubmed_gcn), np.mean(pubmed_kenn_gcn)]
errors = [np.std(pubmed_mlp), np.std(pubmed_kenn_mlp), np.std(pubmed_gcn), np.std(pubmed_kenn_gcn)]
fig, ax = plt.subplots()
ax.bar(xpos, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('test accuracy')
ax.set_xticks(xpos)
ax.set_xticklabels(['MLP', 'KENN_MLP', 'GCN', 'KENN_GCN'])
ax.set_title('Pubmed Test accuracy plot ')
ax.yaxis.grid(True)

# Save the figure and show
plt.ylim(0.6, 1.0)
plt.tight_layout()
plt.savefig('pubmed_error_bars.png')
plt.show()

#pubmed
means = [np.mean(flickr_mlp), np.mean(flickr_kenn_mlp), np.mean(flickr_gcn), np.mean(flickr_kenn_gcn)]
errors = [np.std(flickr_mlp), np.std(flickr_kenn_mlp), np.std(flickr_gcn), np.std(flickr_kenn_gcn)]
fig, ax = plt.subplots()
ax.bar(xpos, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('test accuracy')
ax.set_xticks(xpos)
ax.set_xticklabels(['MLP', 'KENN_MLP', 'GCN', 'KENN_GCN'])
ax.set_title('Flickr Test accuracy plot ')
ax.yaxis.grid(True)

# Save the figure and show
plt.ylim(0.2, 0.6)
plt.tight_layout()
plt.savefig('flickr_error_bars.png')
plt.show()