# plot_metrics.py
import numpy as np
import matplotlib.pyplot as plt

def plot_performance(integrator='rk4'):
    try:
        data = np.load(f"outputs/saved_models/metrics_treino_{integrator}.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print(f"ERRO: O arquivo 'outputs/saved_models/metrics_treino_{integrator}.npy' não foi encontrado!")
        print("Execute o script 'train.py' primeiro para gerar os dados de treino.")
        return

    rewards = data["rewards"]
    epsilons = data["epsilons"]
    episodes = np.arange(1, len(rewards) + 1)

    # Calcula a média móvel (janela de 50 episódios) para suavizar o gráfico
    window_size = 50
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    # Ajusta o vetor de episódios para alinhar com o corte do modo 'valid' da convolução
    smoothed_episodes = episodes[window_size - 1:]

    # Inicializa a figura com proporções acadêmicas (padrão IEEE/ACM de coluna dupla)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

    # Plotagem do Eixo Principal: Recompensa (Curva de Aprendizado)
    # Linha clara ao fundo mostrando o ruído real do treino (opacidade reduzida para evitar ruído visual)
    ax1.plot(episodes, rewards, color='#1f77b4', alpha=0.08, label='Recompensa Real')
    # Linha sólida destacada mostrando a tendência suavizada
    ax1.plot(smoothed_episodes, smoothed_rewards, color='#1f77b4', linewidth=2, label=f'Média Móvel ({window_size} ep.)')
    
    # Limita o eixo Y com base na variação da média móvel para evitar distorções por picos atípicos
    ymin = np.min(smoothed_rewards)
    ymax = np.max(smoothed_rewards)
    margin = (ymax - ymin) * 0.1 if ymax != ymin else 1.0
    ax1.set_ylim(ymin - margin, ymax + margin)
    
    ax1.set_xlabel('Episódios de Treinamento', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Recompensa Acumulada', color='#1f77b4', fontweight='bold', labelpad=10)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Plotagem do Eixo Secundário Compartilhado (Twinx): Decaimento do Epsilon
    ax2 = ax1.twinx()
    ax2.plot(episodes, epsilons, color='#ff7f0e', linestyle='--', linewidth=1.8, label='Exploração ($\epsilon$)')
    ax2.set_ylabel('Taxa de Exploração ($\epsilon$)', color='#ff7f0e', fontweight='bold', labelpad=10)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    # Consolidação de Legendas de Ambos os Eixos
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

    # Título Metodológico e Ajustes Finais
    plt.title(f'Dinâmica de Convergência do Agente: Q-Learning Tabular + {integrator}', 
              fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()

    # Salva uma cópia em alta definição para o documento final do trabalho
    plt.savefig(f"outputs/plots/resultado_convergencia_cartpole_{integrator}.png", bbox_inches='tight', dpi=300)
    print(f"-> Gráfico acadêmico salvo em alta resolução: 'outputs/plots/resultado_convergencia_cartpole_{integrator}.png'")
    
    # Exibe a janela interativa na tela
    plt.show()

if __name__ == "__main__":
    plot_performance('euler')