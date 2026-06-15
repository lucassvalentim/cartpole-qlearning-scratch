# environment.py
import numpy as np
import pygame

# Constantes Físicas do Sistema Purista
GRAVITY = 9.81
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
HALF_LENGTH = 0.5
POLE_MASS_LENGTH = MASS_POLE * HALF_LENGTH

# Configurações de Tela
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
PIXELS_PER_METER = 150

class CartPoleEnv:
    def __init__(self):
        self.x_threshold = 2.4
        self.theta_threshold_radians = np.radians(12.0)
        self.force_mag = 10.0
        self.dt = 0.02
        self.state = None
        self.screen = None
        self.clock = None
        self.reset()

    def reset(self):
        """Reinicia o ambiente com um ruído estocástico sutil."""
        self.state = np.random.uniform(low=-0.02, high=0.02, size=(4,))
        return self.state

    def _cartpole_derivatives(self, state, force):
        """Equações diferenciais resolvidas na Fase 1."""
        x, x_dot, theta, theta_dot = state
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        temp = (force + POLE_MASS_LENGTH * (theta_dot ** 2) * sin_theta) / TOTAL_MASS
        numerator_theta = GRAVITY * sin_theta - cos_theta * temp
        denominator_theta = HALF_LENGTH * (4.0 / 3.0 - (MASS_POLE * (cos_theta ** 2)) / TOTAL_MASS)
        
        theta_double_dot = numerator_theta / denominator_theta
        x_double_dot = temp - (POLE_MASS_LENGTH * theta_double_dot * cos_theta) / TOTAL_MASS
        
        return np.array([x_dot, x_double_dot, theta_dot, theta_double_dot])

    def step(self, action):
        """Avança o tempo usando o integrador de alta ordem Runge-Kutta (RK4)."""
        force = -self.force_mag if action == 0 else self.force_mag
        
        state = self.state
        k1 = self._cartpole_derivatives(state, force)
        k2 = self._cartpole_derivatives(state + (self.dt / 2.0) * k1, force)
        k3 = self._cartpole_derivatives(state + (self.dt / 2.0) * k2, force)
        k4 = self._cartpole_derivatives(state + self.dt * k3, force)
        self.state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        x, x_dot, theta, theta_dot = self.state
        
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        if not done:
            reward = 1.0 - 0.7 * (theta ** 2) - 0.3 * (x ** 2)
        else:
            reward = -20.0
            
        return self.state, reward, done

    def render(self):
        """Renderização gráfica isolada via Pygame."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("CartPole Modularizado - Q-Learning")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        x, _, theta, _ = self.state
        world_center_x = SCREEN_WIDTH // 2
        world_center_y = int(SCREEN_HEIGHT * 0.7)
        
        # Trilho
        pygame.draw.line(self.screen, (150, 150, 150), (0, world_center_y), (SCREEN_WIDTH, world_center_y), 2)
        
        # Carrinho
        cart_pixel_x = int(world_center_x + x * PIXELS_PER_METER)
        cart_rect = pygame.Rect(cart_pixel_x - 40, world_center_y - 20, 80, 40)
        pygame.draw.rect(self.screen, (50, 50, 50), cart_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect, 2)
        
        # Haste
        pole_len_pixels = HALF_LENGTH * 2 * PIXELS_PER_METER
        pole_end_x = int(cart_pixel_x + pole_len_pixels * np.sin(theta))
        pole_end_y = int(world_center_y - pole_len_pixels * np.cos(theta))
        pygame.draw.line(self.screen, (139, 69, 19), (cart_pixel_x, world_center_y), (pole_end_x, pole_end_y), 6)
        
        # Articulação
        pygame.draw.circle(self.screen, (0, 0, 255), (cart_pixel_x, world_center_y), 6)
        
        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.screen is not None:
            pygame.quit()