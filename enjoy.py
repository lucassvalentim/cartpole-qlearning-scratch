# enjoy.py
import sys
import pygame
from core.environment import CartPoleEnv
from core.agent import QLearningAgent

if __name__ == "__main__":
    env = CartPoleEnv()
    agent = QLearningAgent()
    
    # Tenta carregar o cérebro persistido gerado pelo train.py
    try:
        agent.load_policy("outputs/saved_models/q_table_otimizada.npy")
    except FileNotFoundError:
        print("ERRO: O arquivo 'q_table_otimizada.npy' não foi encontrado!")
        print("Por favor, execute o script 'train.py' primeiro para treinar o modelo.")
        sys.exit()
        
    print("\nIniciando demonstração da política ótima estabilizada.")
    print("Feche a janela gráfica para encerrar o programa de testes.")
    
    for demo_ep in range(5):
        continuous_state = env.reset()
        state_idx = agent.discretize_state(continuous_state)
        done = False
        step_count = 0
        
        while not done:
            # Inicializa e renderiza a janela gráfica antes de checar eventos
            env.render()
            
            # Captura eventos de saída nativos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()
            
            # Usa Train=False para desativar comportamento aleatório (Epsilon)
            action = agent.choose_action(state_idx, train=False)
            next_continuous_state, _, done = env.step(action)
            state_idx = agent.discretize_state(next_continuous_state)
            
            step_count += 1
            if step_count > 500:  # Trava nos ~10 segundos ótimos
                break
                
        print(f"Episódio Demonstrativo {demo_ep + 1} -> Tempo de Estabilidade: {step_count} frames.")
        
    env.close()