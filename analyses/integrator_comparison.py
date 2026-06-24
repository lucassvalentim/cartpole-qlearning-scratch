# integrator_comparison.py
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys

# Garante que o diretório raiz está no path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import CartPoleEnv
from core.agent import QLearningAgent

def run_training_experiment(integrator_name, episodes=4000, seed=42):
    # Define a semente global dos dados.
    np.random.seed(seed)
    random.seed(seed)

    env = CartPoleEnv(integrator=integrator_name)
    agent = QLearningAgent()
    
    rewards_history = []
    
    for episode in range(1, episodes + 1):
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
            
        # Decaimento dos hiperparâmetros
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        if agent.alpha > agent.alpha_min:
            agent.alpha *= agent.alpha_decay
            
        rewards_history.append(total_reward)
        
    return rewards_history

def main():
    print("=================== COMPARAÇÃO DE INTEGRADORES ===================")
    episodes = 8000
    window_size = 50
    
    # Executa os experimentos
    print("-> Treinando agente com integrador RK4...")
    rewards_rk4 = run_training_experiment("rk4", episodes=episodes)
    
    print("-> Treinando agente com integrador Euler...")
    rewards_euler = run_training_experiment("euler", episodes=episodes)
    
    # Aplica média móvel
    smoothed_rk4 = np.convolve(rewards_rk4, np.ones(window_size)/window_size, mode='valid')
    smoothed_euler = np.convolve(rewards_euler, np.ones(window_size)/window_size, mode='valid')
    episodes_axis = np.arange(window_size, episodes + 1)
    
    # Certifica que a pasta de plots existe
    os.makedirs("outputs/plots", exist_ok=True)
    
    # ----------------------------------------------------
    # 1. GRÁFICO COMPARATIVO (SOBREPOSTO)
    # ----------------------------------------------------
    plt.figure(figsize=(11, 6), dpi=150)
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    
    plt.plot(episodes_axis, smoothed_rk4, label='Runge-Kutta 4ª Ordem (RK4)', color='#1f77b4', linewidth=2.5)
    plt.plot(episodes_axis, smoothed_euler, label='Euler (1ª Ordem)', color='#d62728', linewidth=2.5)
    
    plt.title("Estudo Comparativo de Integradores Numéricos: RK4 vs. Euler", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Episódios de Treinamento", fontweight='bold', labelpad=10)
    plt.ylabel(f"Recompensa Média Móvel (Janela de {window_size} ep.)", fontweight='bold', labelpad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', framealpha=0.9, edgecolor='black')
    plt.xlim(window_size, episodes)
    
    comparison_path = "outputs/plots/comparacao_integradores.png"
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> Sucesso: Gráfico comparativo salvo em '{comparison_path}'")
    
    # ----------------------------------------------------
    # 2. GRÁFICO INDIVIDUAL: RK4
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(episodes_axis, smoothed_rk4, color='#1f77b4', linewidth=2)
    plt.title("Dinâmica de Convergência - Integrador RK4", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Episódios de Treinamento", labelpad=8)
    plt.ylabel(f"Recompensa Média Móvel ({window_size} ep.)", labelpad=8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(window_size, episodes)
    
    rk4_path = "outputs/plots/convergencia_rk4.png"
    plt.savefig(rk4_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> Sucesso: Gráfico individual RK4 salvo em '{rk4_path}'")
    
    # ----------------------------------------------------
    # 3. GRÁFICO INDIVIDUAL: EULER
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(episodes_axis, smoothed_euler, color='#d62728', linewidth=2)
    plt.title("Dinâmica de Convergência - Integrador Euler", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Episódios de Treinamento", labelpad=8)
    plt.ylabel(f"Recompensa Média Móvel ({window_size} ep.)", labelpad=8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(window_size, episodes)
    
    euler_path = "outputs/plots/convergencia_euler.png"
    plt.savefig(euler_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> Sucesso: Gráfico individual Euler salvo em '{euler_path}'")
    
    print("\n==================================================")
    print("Experimento finalizado com sucesso!")
    print("Os gráficos individuais e comparativo estão prontos.")
    print("==================================================")

if __name__ == "__main__":
    main()
