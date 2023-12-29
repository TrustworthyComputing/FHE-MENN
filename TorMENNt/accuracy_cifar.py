import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.special import kl_div
from sklearn.metrics import fbeta_score
from sklearn.metrics import mutual_info_score

categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = 10
exit_labels = ['Exit0', 'Exit1', 'Exit2']
NUM_EXITS = 3
FNAME = "cifar-vgg"

def accuracy(correct, entropies, target=None, tclass=-1, thresh_mod =1, subsample = -1, verbose = True):
    
    if target is not None and tclass >= 0:
        num_samples = np.count_nonzero((target == tclass))
        num_subsamples = num_samples
        if subsample >0:
            rng = np.random.default_rng()
            num_subsamples = min(num_samples, subsample)
            samples = rng.integers(num_samples, size=num_subsamples)
            
        new_correct = np.zeros((NUM_EXITS, num_subsamples))
        new_entropies = np.zeros((NUM_EXITS, num_subsamples))

        for ei in range(NUM_EXITS):
            tmpc = correct[ei][(target == tclass)]
            tmpe = entropies[ei][(target == tclass)] 
            if subsample > 0:
                new_correct[ei] = tmpc[samples]
                new_entropies[ei] = tmpe[samples]
                #print(tmpc[samples])
                #print(f"Shapes: {tmpc.shape} {new_correct.shape} {samples}")
            else:
                new_correct[ei] = tmpc
                new_entropies[ei] = tmpe
        correct = new_correct
        entropies = new_entropies
    thresh = scipy.stats.entropy(np.ones(NUM_CLASSES)/NUM_CLASSES)/2
    thresh *= thresh_mod
    #default exit last
    exits = np.ones(len(correct[0]))*(NUM_EXITS-1)
    # Work backwards to find first exit that had an entropy below thresh
    for ei in range(NUM_EXITS-2, -1, -1):
        mask = (entropies[ei] < thresh)
        exits = np.where(mask, ei, exits)

    percisions = []
    recalls = []
    total_correct = 0
    for ei in range(NUM_EXITS):
        num_exit = np.count_nonzero(exits == ei)
        exit_correct = correct[ei][exits == ei]
        num_correct = np.sum(exit_correct)
        if verbose:
            print(f"Exit {ei}: {num_correct}/{num_exit} = {100*num_correct/num_exit}%")
        
        recalls.append(num_exit)
        percisions.append(num_correct)
        total_correct += num_correct
    if verbose:
        print(f"Overall Accuracy: {total_correct}/{len(correct[0])} = {100*total_correct/len(correct[0])}%")
    return percisions, recalls, exits

def unwrap(array, b_reshape=False):
    new_array = array[0].flatten()
    for i in range(1,len(array)):
        new_array = np.concatenate((new_array, array[i].flatten()))
    return new_array

def load_and_run():
    
    base = FNAME + "/npy/entropies"
    corr_tag = 'fcorrect'
    entr_tag = 'fentropy'
   
    rng = np.random.default_rng()
    split = np.arange(10000)
    rng.shuffle(split)
    test = split[:]
    attack = split[:]

    for ei in range(NUM_EXITS):
        data = np.load(f"{base}{ei}.npz", allow_pickle=True)
        if ei != 0:
            correct = np.vstack((correct, unwrap(data[corr_tag])))
            entropies = np.vstack((entropies, unwrap(data[entr_tag])))
            print(f"Single Exit {ei}: {100*np.sum(correct[ei])/len(correct[ei])}\%")
        else:
            correct = unwrap(data[corr_tag], True)
            entropies = unwrap(data[entr_tag])
            target = unwrap(data['target'])
            print(f"Single Exit {ei}: {100*np.sum(correct)/len(correct)}\%")

    print(entropies.shape, correct.shape, target.shape)

    for cl in range(NUM_CLASSES):
        print(f"Class {categories[cl]}")
        percision_cl, recall_cl, exits = accuracy(correct[:,test], entropies[:,test], target[test], cl)

        if cl != 0:
            percisions = np.vstack((percisions, np.array(percision_cl).flatten()))
            recalls = np.vstack((recalls, np.array(recall_cl).flatten()))
        else:
            percisions = np.array(percision_cl).flatten()
            recalls = np.array(recall_cl).flatten()
    
    display_radar_chart(np.cumsum(recalls, axis=1).T/1000, 'Cumlative Recall',  categories, exit_labels, FNAME+"recall.png")
    display_radar_chart(recalls.T/1000, 'Proability Exit is Taken',  categories, exit_labels, FNAME+"pexit.png")
    display_radar_chart(np.array([np.sum(percisions, axis=1).T/1000]), 'Class Based Percision',  categories, ['Per Class Accuaracy'], FNAME+"percision.png")
    np.savetxt(FNAME+"/cumrecalls.txt", np.cumsum(recalls, axis=1))
    np.savetxt(FNAME+"/accuracy.txt", np.array([np.sum(percisions, axis=1)]))

    for ti in range(7, 14):
        print(f"Thresh: {ti}")
        pecision_all, recall_all, exited = accuracy(correct, entropies, thresh_mod = ti/10, verbose=True)
        print()

    pecision_all, recall_all, exited = accuracy(correct, entropies, verbose=False)


    num_iter = 100
    num_pics = 1
    num_top = 1
    side_channel_onehot_exit, inc, attack_acc, model_safety_acc = \
            kl_div_per_class(exited[attack], target[attack], recalls, num_iter, num_pics, num_top) 

    print(attack_acc.T.shape)

    plt.show()
    plt.subplot(polar=False)
    plt.xlabel("Number of Pictures in Batch")
    plt.ylabel("Attacker Accuracy")
    plt.title("Cifar-10 Side Channel Effectiveness on User Data")
    for j in range(NUM_CLASSES):
        plt.plot(np.arange(1, num_iter+1), (attack_acc.T)[j], label=categories[j], linestyle='--' )
    plt.plot(np.arange(1, num_iter+1), attack_acc.T[-1], label="Overall", linestyle='-', linewidth=2.0, color='black' )
    # show legend
    plt.legend()

    # show graph
    plt.savefig(f"imgs/{FNAME}-attack1.png", dpi =400)
    np.save(FNAME + '/attack_acc.npy', attack_acc[0])
    plt.show()

    plt.xlabel("Number of Pictures in Batch")
    plt.ylabel("Attacker Accuracy")
    plt.title("Cifar-10 Side Channel Effectiveness on User Data")
    for j in range(NUM_CLASSES):
        plt.plot(np.arange(1, num_iter+1), (model_safety_acc.T)[j], label=categories[j], linestyle='--' )
    plt.plot(np.arange(1, num_iter+1), model_safety_acc.T[-1], label="Overall", linestyle='-', linewidth=2.0, color='black' )
    # show legend
    plt.legend()

    # show graph
    #plt.show()
    plt.savefig(f"imgs/{FNAME}-attack2.png", dpi =400)


def display_radar_chart(values, title, _categories, labels, savename=None):       
    num_classes = len(values[0])
    num_exits = len(values)
    print(values.shape, num_classes, num_exits)
    
    _categories = [*_categories, _categories[0]]
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=num_classes+1)
    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    for i in range(num_exits):
        plot_value = [*values[i], values[i][0]]
        plt.plot(label_loc, plot_value, label=labels[i], linewidth=3)
    
    plt.title(title, size=20)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels="")
    #plt.legend()
    #plt.show()
    plt.savefig(f"imgs/{savename}", dpi =400)

def kl_div_per_class(side_channel_exit, labels, indiv_recalls, num_iter=100, num_pics=1, num_top=1): 
    num_samples = len(labels)

    attack_acc = np.zeros((num_top, num_iter,NUM_CLASSES+1))
    model_safety_acc = np.zeros((num_top, num_iter,NUM_CLASSES+1))   
    #print(side_channel_exit.shape) 
    side_channel_onehot_exit = get_one_hot(side_channel_exit.astype(int), NUM_EXITS)

    for npc in range(num_pics, num_pics+num_iter, 1):

        num_subsamples = int(num_samples/NUM_CLASSES) #int(num_samples/num_pics/num_classes)+100

        divergences = np.ones((NUM_CLASSES*num_subsamples, NUM_CLASSES))
        recall_per_class = np.zeros((NUM_CLASSES))
        side_channel_average_onehot = []
        sample_ilabels = [] #np.zeros((num_subsamples*num_classes))
        
        #combine picture results
        k=0
        for j in range(NUM_CLASSES):
            cat_one_hot = (side_channel_onehot_exit[labels == j])
            num_cat_samples = len(cat_one_hot)
            for ks in range(num_subsamples):
                idx = np.random.randint(num_cat_samples, size=npc+1)
                side_channel_average_onehot.append(np.average(cat_one_hot[idx], axis=0))
                sample_ilabels.append(j) 

        recall_per_class = (indiv_recalls)

        #predict classes
        for k in range(num_subsamples*NUM_CLASSES):
            for j in range(NUM_CLASSES):
                divergences[k][j] = np.sum(kl_div(side_channel_average_onehot[k], recall_per_class[j]))
                #print(len(side_channel_average_onehot[k]), len(recall_per_class[j]))
            #print(divergences[k])

        attack_ilbl = np.argmin(divergences, axis=1)

        #print(attack_acc.shape)
        for j in range(NUM_CLASSES):
            #How often the attacker is correct based on a given class
            cat_lbl = np.array(sample_ilabels, dtype=int)[attack_ilbl == j]
            if(len(cat_lbl) > 10):
                incorrect = np.clip(np.abs(j - cat_lbl), 0, 1) 
                attack_acc[0][npc-num_pics][j] = (1-np.average(incorrect))
            else:
                attack_acc[0][npc-num_pics][j] = float("NAN")

            #given the user sends class x, how often is it guessed
            cat_attack_ilbl = attack_ilbl[np.array(sample_ilabels, dtype=int)==j]
            if (len(cat_attack_ilbl) > 10):
                incorrect = np.clip(np.abs(cat_attack_ilbl - j), 0, 1) 
                model_safety_acc[0][npc-num_pics][j] = (1-np.average(incorrect))
            else:
                model_safety_acc[0][npc-num_pics][j] = float("NAN")

        #and add overall accuracies
        incorrect = np.clip(np.abs(attack_ilbl - sample_ilabels), 0, 1) 
        attack_acc[0][npc-num_pics][-1] = (1-np.average(incorrect))
        model_safety_acc[0][npc-num_pics][-1] = (1-np.average(incorrect))
    
    return side_channel_onehot_exit, incorrect, attack_acc, model_safety_acc

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


load_and_run()
