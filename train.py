# train.py
import numpy as np
from core.environment import CartPoleEnv
from core.agent import QLearningAgent
import random

# Define a semente global dos dados.
SEMENTE_CIE_DADOS = 42
np.random.seed(SEMENTE_CIE_DADOS)
random.seed(SEMENTE_CIE_DADOS)

if __name__ == "__main__":
    env = CartPoleEnv()
    agent = QLearningAgent()
    
    num_episodes = 4000
    print(f"Iniciando treinamento puramente matemático de {num_episodes} episódios...")
    
    # Estruturas para registrar os dados acadêmicos que usaremos na plotagem de gráficos
    reward_history = []
    epsilon_history = []
    
    for episode in range(1, num_episodes + 1):
        continuous_state = env.reset()
        state_idx = agent.discretize_state(continuous_state)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state_idx, train=True)
            next_continuous_state, reward, done = env.step(action)
            next_state_idx = agent.discretize_state(next_continuous_state)
            
            agent.learn(state_idx, action, reward, next_state_idx)
            
            state_idx = next_state_idx
            total_reward += reward
            
        # Armazena os dados históricos do episódio
        reward_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        
        # Decaimento dos hiperparâmetros lineares/exponenciais
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        if agent.alpha > agent.alpha_min:
            agent.alpha *= agent.alpha_decay
            
        if episode % 500 == 0 or episode == 1:
            print(f"Episódio: {episode:<5} | Epsilon: {agent.epsilon:.3f} | Alpha: {agent.alpha:.3f} | Recompensa: {total_reward:.2f}")

    print("\nTreinamento Finalizado!")
    
    # SALVAMENTO MODULAR
    agent.save_policy("outputs/saved_models/q_table_otimizada.npy")
    
    # Salva os dados analíticos das métricas para a Fase de Gráficos
    dados_graficos = {
        "rewards": reward_history,
        "epsilons": epsilon_history
    }
    np.save("outputs/saved_models/metrics_treino.npy", dados_graficos)
    print("-> Métricas de treinamento exportadas para 'outputs/saved_models/metrics_treino.npy'. ready para plotagem.")