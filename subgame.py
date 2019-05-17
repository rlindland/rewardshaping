import numpy as np

for i in range(16):
	strat = [i%2, i//2%2, i//2//2%2, i//2//2//2%2]

	wow = []
	for ii in range(16):
		strat2 = [ii%2, ii//2%2, ii//2//2%2, ii//2//2//2%2]
		vars = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
		for x in range(4):
			vars[x][strat[x] + 2*strat2[x]] += -0.9

		wowow = np.linalg.solve(np.array(vars), np.array([3,0,0,1]))


		wow.append(wowow)

	if np.all(np.array(wow) <= np.array(wow[i])):
		print(strat)

	# print(wow)

