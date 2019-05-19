import numpy as np

wow = []
strat = []

#Chicken - [5,4,6,0]
#row_rewards = [0,-1,1,-5]
#Battle of the Sexes - [3,0,0,2]
row_rewards = [3,0,0,2]
#row_rewards = [3,0,4,1]

#Chicken - [5,6,4,0]
#col_rewards = [0,1,-1,-5]
#Battle of the Sexes - [2,0,0,3]
col_rewards = [2,0,0,3]
#col_rewards = [3,4,0,1]
for ii in range(16):
	strat2 = [ii%2, ii//2%2, ii//2//2%2, ii//2//2//2%2]
	vars = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
	for x in range(4):
		if x==0 or x==3:
			new_x=x
		else:
			new_x=3-x
		vars[x][strat2[x] + 2*strat2[new_x]] += -0.9

	wowow = np.linalg.solve(np.array(vars), np.array(row_rewards))


	strat2 = [ii%2, ii//2%2, ii//2//2%2, ii//2//2//2%2]
	vars = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
	for x in range(4):
		if x==0 or x==3:
			new_x=x
		else:
			new_x=3-x
		vars[x][strat2[x] + 2*strat2[new_x]] += -0.9

	wowow2 = np.linalg.solve(np.array(vars), np.array(col_rewards))

	toAdd = True
	check = [(strat2[0],strat2[0]), (strat2[1], strat2[2]), (strat2[2],strat2[1]), (strat2[3], strat2[3])]
	for row, col in check:
		index = 2*row+col
		row_check_ind = 2*((row+1)%2)+col
		col_check_ind = 2*row+(col+1)%2
		if (wowow[index]<wowow[row_check_ind]):toAdd=False
		if (wowow2[index]<wowow2[col_check_ind]): toAdd = False
	if toAdd:
		print("strategy: "+str(strat2))
		print("Q Matrix initialization for ROW PLAYER")
		Q = {}
		for your_state in [0,1]:
		    for their_state in [0,1]:
		        for move in [0,1]:
		            Q[your_state, their_state]={}
		            Q[(your_state, their_state)][move]=wowow[2*move+strat2[their_state*2+your_state]]
		            print("Q[("+str(your_state)+","+str(their_state)+")]["+str(move)+"]="+str(Q[(your_state,their_state)][move]))
		print()
