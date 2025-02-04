import numpy as np #Cette ligne permet d'importer le jeu de donne numpy / This line imports the numpy data set

grid_size = 2 #Taille de la grille pa exemple 2*2 / Grid size for example 2*2
rewards = np.zeros(grid_size * grid_size) #tableau de récompense en fonction de la taille de la grille / reward table based on the grid size
rewards[1] = 10 #la case numéro 1 du tableau de récompense vaut 10 points / cell number 1 of the reward table is worth 10 points
actions = [0, 1, 2, 3] #tableau des actions 0: haut, 1: droite, 2: bas, 3: gauche / action table 0: top, 1: right, 2: bottom, 3: left 
Q_table = np.zeros((grid_size * grid_size, len(actions))) #Q_table chaque ligne represente les états et chaque colonne represente les actions / Q_table each row represents the states and each column represents the actions
learning_rate = 0.1 #Vitesse d'aprentissage / Learning speed
discount_factor = 0.9 #Importance des recompences futures / Importance of future rewards
epsilon = 0.8 #Taux d'exploration / Exploration rate
episodes = 100 #Nombres d'episode / Numbers of episodes

#Cette fonction permet de choisir l'action / This function allows you to choose the action
def choose_action(state, actions):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q_table[state])

#Cette fonction permet de deplacer l'agent sur une autre case en fonction de l'action choisie / This function allows you to move the agent to another case depending on the chosen action    
def step(state, action):
    if action == 0:
        next_state = state - grid_size if state - grid_size >=0 else state
    elif action == 1:
        next_state = state + 1 if state % grid_size != grid_size - 1 else state
    elif action == 2:
        next_state = state + grid_size if state + grid_size < grid_size * grid_size else state
    elif action == 3:
        next_state = state - 1 if state % grid_size != 0 else state
    return next_state

#L'agent commence sur la première case et doit atteindre la deuxieme et il a en total 100 tentatives / The agent starts on the first square and must reach the second and has a total of 100 attempts
for episode in range(episodes):
    state = 0
    done = False
    while not done:
        action = choose_action(state, actions)
        next_state = step(state, action)
        reward = rewards[next_state]
        Q_table[state, action] = Q_table[state, action] + learning_rate*(reward + discount_factor*np.max(Q_table[next_state]) - Q_table[state, action])
        state = next_state
        if reward == 10:
            done = True

#Ces 2 lignes permet d'afficher le q_table apres l'entrainement / These 2 lines allow to display the q_table after training
print("Q_table après l'entrainement:")
print(Q_table)

        
