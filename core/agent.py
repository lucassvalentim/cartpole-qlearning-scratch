# agent.py
import numpy as np
import random

class QLearningAgent:
    def __init__(
        self,
        alpha = 0.3,
        alpha_decay = 0.995,
        alpha_min = 0.01,
        epsilon = 1.0,   
        epsilon_decay = 0.998,  
        epsilon_min = 0.01,
        gamma = 0.99
    ):
        # Hiperparâmetros Iniciais
        self.alpha = alpha    
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        # Definição das Malhas Não-Lineares de Discretização (Foco em 0 graus)
        self.bins_x = np.array([-1.5, -0.5, 0.5, 1.5])            
        self.bins_x_dot = np.array([-1.5, -0.5, 0.5, 1.5])        
        
        base_points_theta = np.linspace(-1, 1, 9)
        self.bins_theta = np.sign(base_points_theta) * (np.abs(base_points_theta) ** 1.8) * np.radians(12.0)
        
        base_points_omega = np.linspace(-1, 1, 7)
        self.bins_theta_dot = np.sign(base_points_omega) * (np.abs(base_points_omega) ** 1.5) * 2.0
        
        # Tabela Q Inicializada Zera (5x5x10x8x2)
        self.q_table = np.zeros((5, 5, 10, 8, 2))

    def discretize_state(self, continuous_state):
        """Converte o vetor contínuo do ambiente em índices discretos estruturados."""
        x, x_dot, theta, theta_dot = continuous_state
        state_idx = (
            np.digitize(x, self.bins_x),
            np.digitize(x_dot, self.bins_x_dot),
            np.digitize(theta, self.bins_theta),
            np.digitize(theta_dot, self.bins_theta_dot)
        )
        return state_idx

    def choose_action(self, state_idx, train=True):
        """Estratégia Epsilon-Greedy."""
        if train and random.random() < self.epsilon:
            return random.randint(0, 1) 
        else:
            return np.argmax(self.q_table[state_idx]) 

    def learn(self, state_idx, action, reward, next_state_idx):
        """Atualização temporal clássica por Equação de Bellman."""
        best_next_action = np.argmax(self.q_table[next_state_idx])
        td_target = reward + self.gamma * self.q_table[next_state_idx][best_next_action]
        td_error = td_target - self.q_table[state_idx][action]
        self.q_table[state_idx][action] += self.alpha * td_error

    def save_policy(self, filename="q_table.npy"):
        """Salva a matriz de conhecimento calculada em um arquivo binário comprimido."""
        np.save(filename, self.q_table)
        print(f"-> Política de controle exportada com sucesso para: {filename}")

    def load_policy(self, filename="q_table.npy"):
        """Carrega uma matriz pré-treinada para a memória ativa do agente."""
        self.q_table = np.load(filename)
        print(f"-> Política de controle '{filename}' carregada na memória do agente.")