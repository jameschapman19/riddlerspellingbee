from collections import Counter
import numpy as np
import pandas as pd

f = open('enable1.txt', 'r')
wordlist = f.read().splitlines()
f.close()
wordlist = [word for word in wordlist if len(word) >= 4]

wordnumber = 0
counts = {}
for word in wordlist:
    counts[wordnumber] = Counter(word)  # Counter({'l': 2, 'H': 1, 'e': 1, 'o': 1})
    wordnumber += 1
    if wordnumber % 1000 == 0:
        print(wordnumber)

#generate a tabular version of each word (number of words x 26 letters)
df = pd.DataFrame.from_dict(counts)
lettercounts = df.values
#save the meaning of each column - the order gets confused
letter_index = df.index.values
lettercounts = np.transpose(lettercounts)
lettercounts[np.isnan(lettercounts)] = 0

# filter for not containing s
s = lettercounts[:, 7]
lettercounts = lettercounts[s == 0]

#filter for having no more than 7 unique letters
seven_max = np.count_nonzero(lettercounts, axis=1)
lettercounts = lettercounts[seven_max < 8]

# calculate points for each word
score = np.sum(lettercounts, axis=1)
#if pangram get 7 bonus
score[np.count_nonzero(lettercounts, axis=1) == 7] += 7
# expand to give score conditional on that letter being in centre of bee game (26 dimensional)
score = (lettercounts > 0) * score[:, None]

# filter pangram words
pangrams = lettercounts[np.count_nonzero(lettercounts, axis=1) == 7]
# extract the letters used in the pangrams
letter_combinations = np.argwhere(pangrams > 0)[:,1].reshape(-1,7)

# iterate through combinations
combination_scores = np.zeros((len(letter_combinations), 7))

# apply to score
n = 0
for letter_combination in letter_combinations:
    combination_scores[n, :] = np.take(score[np.where(
        ~lettercounts[:, [i not in letter_combination for i in range(lettercounts.shape[1])]].any(axis=1))[0], :].sum(
        axis=0), letter_combination)
    n += 1
    if n % 100 == 0:
        print(n)

max_index = np.unravel_index(combination_scores.argmax(), combination_scores.shape)
min_index = np.unravel_index(combination_scores.argmin(), combination_scores.shape)

#print results

# print BEST combo
best_letters = np.take(letter_index, letter_combinations[max_index[0]])
print("letters = " + str(best_letters))

# print golden letter
golden_letter = letter_index[letter_combinations[max_index[0]][max_index[1]]]
print("golden_letter = " + str(golden_letter))

# print score
print("score = " + str(combination_scores[max_index]))

# print WORST combo
best_letters = np.take(letter_index, letter_combinations[min_index[0]])
print("letters = " + str(best_letters))

# print golden letter
golden_letter = letter_index[letter_combinations[min_index[0]][min_index[1]]]
print("golden_letter = " + str(golden_letter))

# print score
print("score = " + str(combination_scores[min_index]))


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots()
ax.hist(combination_scores)
fig.tight_layout()
plt.show()
np.save('combination_scores.npy', combination_scores)
