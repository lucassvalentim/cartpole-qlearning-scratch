# hyperparameter_study.py
import numpy as np
import random
import matplotlib.pyplot as plt
from core.environment import CartPoleEnv
from core.agent import QLearningAgent

def run_experiment(alpha, gamma, episodes=4000, seed=42):
    # Garante o determinismo absoluto para o experimento ser justo
    np.random.seed(seed)
    random.seed(seed)
    
    env = CartPoleEnv()
    
    # Criamos o agente injetando os parâmetros do teste atual
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon_decay=0.999)
    
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        state_idx = agent.discretize_state(state)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state_idx, train=True)
            next_state, reward, done = env.step(action)
            next_state_idx = agent.discretize_state(next_state)
            
            agent.learn(state_idx, action, reward, next_state_idx)
            
            state_idx = next_state_idx
            total_reward += reward
            
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        agent.alpha = max(agent.alpha_min, agent.alpha * agent.alpha_decay)
        
        rewards_history.append(total_reward)
        
    return rewards_history

def main():
    print("=================== ESTUDO DE HIPERPARAMETROS ===================")
    
    # Definição dos cenários de teste para o gráfico comparativo
    scenarios = [
        ("Cenário A: Padrão (α=0.2, γ=0.99)", 0.2, 0.99),
        ("Cenário B: Aprendizado Rápido (α=0.5, γ=0.99)", 0.5, 0.99),
        ("Cenário C: Visão de Curto Prazo (α=0.2, γ=0.80)", 0.2, 0.80),
        ("Cenário D: Lento/Conservador (α=0.02, γ=0.99)", 0.02, 0.99)
    ]
    
    # Estilização do gráfico para publicação científica
    plt.figure(figsize=(11, 6), dpi=150)
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    
    window_size = 50  # Janela da média móvel para suavizar as linhas
    
    for name, alpha, gamma in scenarios:
        print(f"\n-> Treinando {name}...")
        rewards = run_experiment(alpha=alpha, gamma=gamma, episodes=4000)
        
        # Aplica a média móvel nos dados coletados
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        episodes_axis = np.arange(window_size, len(rewards) + 1)
        
        plt.plot(episodes_axis, smoothed_rewards, label=name, linewidth=2)
    
    # Elementos visuais do gráfico do artigo
    plt.title("Análise Comparativa de Hiperparâmetros: Taxa de Convergência", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Episódios de Treinamento", fontweight='bold', labelpad=10)
    plt.ylabel("Recompensa Média Móvel (Janela de 50 ep.)", fontweight='bold', labelpad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', framealpha=0.9, edgecolor='black')
    plt.xlim(window_size, 2000)
    
    # Salva o arquivo em alta definição (perfeito para incluir no Word ou LaTeX)
    plt.savefig("outputs/plots/estudo_hiperparametros_cartpole.png", bbox_inches='tight', dpi=300)
    print("\n==================================================")
    print("-> SUCESSO: Gráfico salvo como 'outputs/plots/estudo_hiperparametros_cartpole.png'")
    print("==================================================")
    plt.show()

if __name__ == "__main__":
    main()