# play_manual.py
import sys
import pygame
from core.environment import CartPoleEnv

def main():
    # Inicializa o ambiente físico reaproveitado
    env = CartPoleEnv()
    env.force_mag = 5
    env.render()
    
    running = True
    action = 0  # Começa empurrando para a esquerda por padrão
    
    while running:
        # Reinicia o ambiente físico para um novo round
        env.reset()
        score_frames = 0
        print("\n[Status] Sistema Iniciado/Reiniciado! Tente equilibrar...")
        
        # Loop contínuo do gameplay (ignora o 'done' clássico)
        while True:
            env.render()
            score_frames += 1
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    elif event.key == pygame.K_r:
                        break
            
            if not running:
                break
                
   
            if any(event.type == pygame.KEYDOWN and event.key == pygame.K_r for event in pygame.event.get()):
                break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = 0
            elif keys[pygame.K_RIGHT]:
                action = 1
            
            _, _, _ = env.step(action)
            
            if score_frames % 50 == 0:
                print(f"Tempo ativo no round atual: {score_frames} frames", end="\r")

        if not running:
            break

    env.close()
    print("\n\nSimulador manual encerrado. Pronto para a apresentação!")

if __name__ == "__main__":
    main()