

import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

dataset='imdb'

from data.loader import DataLoader
dl = DataLoader()
train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
train_ground, val_ground, test_ground, _, _, _ = dl.load_data(dataset=dataset)

train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
train_ground, val_ground, test_ground = dl.load_data_synt()

from program_synthesis.heuristic_generator import HeuristicGenerator

hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix, val_ground, train_ground, b=0.25)
hg.run_synthesizer(max_cardinality=1, idx=None, keep=3, model='lr')

from program_synthesis.synthesizer import Synthesizer
syn = Synthesizer(val_primitive_matrix, val_ground, b=0.25)

heuristics, feature_inputs = syn.generate_heuristics('lr', 1)
print("Total Heuristics Generated: ", np.shape(heuristics)[1])

optimal_betas = syn.find_optimal_beta(heuristics[0], val_primitive_matrix, feature_inputs[0], val_ground)
plt.hist(optimal_betas, range=(0,0.5));
plt.xlabel('Beta Values');
#plt.show();

top_idx = hg.prune_heuristics(heuristics, feature_inputs, keep=3)
print('Features chosen heuristics are based on: ', top_idx)

from program_synthesis.verifier import Verifier
verifier = Verifier(hg.L_train, hg.L_val, val_ground, has_snorkel=False)

verifier.train_gen_model()
verifier.assign_marginals()

#plt.hist(verifier.train_marginals); plt.title('Training Set Probabilistic Labels');
#plt.show()

#plt.hist(verifier.val_marginals); plt.title('Validation Set Probabilistic Labels');
#plt.show();
feedback_idx = verifier.find_vague_points(gamma=0.1,b=0.25)
print('Percentage of Low Confidence Points: ', np.shape(feedback_idx)[0]/float(np.shape(val_ground)[0]))

validation_accuracy = []
training_accuracy = []
validation_coverage = []
training_coverage = []

training_marginals = []
idx = None

hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix,
                        val_ground, train_ground,
                        b=0.25)
plt.figure(figsize=(12, 6));
for i in range(3, 26):
    if (i - 2) % 5 == 0:
        print(
        "Running iteration: ", str(i - 2))
    # Repeat synthesize-prune-verify at each iterations
    if i <= 3:
        hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='lr')
    else:
        hg.run_synthesizer(max_cardinality=1, idx=idx, keep=1, model='lr')
    hg.run_verifier()

    # Save evaluation metrics
    va, ta, vc, tc = hg.evaluate()
    validation_accuracy.append(va)
    training_accuracy.append(ta)
    training_marginals.append(hg.vf.train_marginals)
    validation_coverage.append(vc)
    training_coverage.append(tc)

    # Plot Training Set Label Distribution
    if i <= 8:
        plt.subplot(2, 3, i - 2)
        plt.hist(training_marginals[-1], bins=10, range=(0.0, 1.0));
        plt.title('Iteration ' + str(i - 2));
        plt.xlim([0.0, 1.0])
        plt.ylim([0, 825])

    # Find low confidence datapoints in the labeled set
    hg.find_feedback()
    idx = hg.feedback_idx

    # Stop the iterative process when no low confidence labels
    if idx == []:
        break
plt.tight_layout()
plt.show()

plt.hist(training_marginals[-1], bins=10, range=(0.0,1.0));
plt.title('Final Distribution');

print("Program Synthesis Train Accuracy: ", training_accuracy[-1])
print("Program Synthesis Train Coverage: ", training_coverage[-1])
print("Program Synthesis Validation Accuracy: ", validation_accuracy[-1])

filepath = './data/' + dataset
np.save(filepath+'_reef.npy', training_marginals[-1])