import numpy as np
import time
import argparse
from PIL import Image, ImageDraw
from mss import mss
import matplotlib.pyplot as plt
import pyautogui
import cv2  # OpenCV library for image processing
import os

# Configurações iniciais
screen_region = {"top": 150, "left": 100, "width": 539, "height": 539}  # Ajuste da área do tabuleiro
grid_rows, grid_cols = 8, 8
mouse_offset = 4  # Offset de 4 pixels

# Variável para controlar o delay de cada jogada
move_delay = .0.5  # Delay de 1 segundo entre as jogadas

# Diretório onde os kernels (templates) estão armazenados
template_dir = 'templates'  # Crie uma pasta chamada 'templates' no mesmo diretório do script

# Dicionário que mapeia tipos de peças aos seus templates (incluindo variações)
template_groups = {
    'circulo_oval': ['circulo_oval.png', 'circulo_oval_2.png', 'circulo_oval_3.png'],
    'hexagono_horizontal': ['hexagono_horizontal.png', 'hexagono_horizontal_2.png', 'hexagono_horizontal_3.png'],
    'hexagono_vertical': ['hexagono_vertical.png', 'hexagono_vertical_2.png'],
    'quadrado_chanfrado': ['quadrado_chanfrado.png', 'quadrado_chanfrado_2.png','quadrado_chanfrado_3.png'],
    'hexagono_irregular': ['hexagono_irregular.png', 'hexagono_irregular_2.png'],
    'heptagono_rosa': ['heptagono_rosa.png', 'heptagono_rosa_2.png']
}

# Carregar os templates agrupados por tipo de peça e calcular as médias de cores
def load_templates():
    templates = {}
    template_colors = {}
    for idx, (piece_type, filenames) in enumerate(template_groups.items()):
        templates[idx] = []
        template_colors[idx] = []
        for filename in filenames:
            path = os.path.join(template_dir, filename)
            if not os.path.exists(path):
                print(f"Aviso: Template '{filename}' não encontrado no diretório '{template_dir}'.")
                continue
            template_image = cv2.imread(path, cv2.IMREAD_COLOR)
            gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            templates[idx].append(gray_template)
            # Calcular a média de cor do template
            avg_color = cv2.mean(template_image)[:3]  # Ignorar canal alfa se existir
            template_colors[idx].append(np.array(avg_color))
    return templates, template_colors

# Função para capturar e desenhar o grid e IDs
def capture_and_draw_grid_with_ids(region, block_labels):
    with mss() as sct:
        screenshot = sct.grab(region)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        draw = ImageDraw.Draw(img)
        block_height = region["height"] // grid_rows
        block_width = region["width"] // grid_cols

        # Desenhar o grid e IDs dos blocos
        for row in range(grid_rows):
            for col in range(grid_cols):
                x = col * block_width
                y = row * block_height
                draw.rectangle([(x, y), (x + block_width, y + block_height)], outline="red", width=2)
                # Exibir o ID do bloco no centro de cada célula
                label = str(block_labels[row, col])
                text_x = x + block_width // 2 - 5
                text_y = y + block_height // 2 - 10
                draw.text((text_x, text_y), label, fill="white")
    return img

# Função para capturar o tabuleiro e processar os blocos
def capture_board(region):
    with mss() as sct:
        screenshot = sct.grab(region)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    return np.array(img)

def process_image(image, templates, template_colors):
    block_height = image.shape[0] // grid_rows
    block_width = image.shape[1] // grid_cols
    blocks = [[image[row * block_height:(row + 1) * block_height,
                     col * block_width:(col + 1) * block_width]
               for col in range(grid_cols)] for row in range(grid_rows)]

    block_labels = np.zeros((grid_rows, grid_cols), dtype=int)
    best_scores = np.zeros((grid_rows, grid_cols))
    second_best_matches = np.zeros((grid_rows, grid_cols), dtype=int)
    second_best_scores = np.zeros((grid_rows, grid_cols))
    for i, row in enumerate(blocks):
        for j, block in enumerate(row):
            # Converter o bloco para escala de cinza e cor
            gray_block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
            color_block = cv2.cvtColor(block, cv2.COLOR_RGB2BGR)
            avg_color_block = np.array(cv2.mean(color_block)[:3])

            best_match = None
            best_score = -np.inf
            second_best_match = None
            second_best_score = -np.inf

            # Iterar sobre cada tipo de peça
            for label, template_list in templates.items():
                max_score_for_type = -np.inf
                # Iterar sobre as variações de cada tipo
                for idx, template in enumerate(template_list):
                    # Redimensionar o template para o tamanho do bloco, se necessário
                    resized_template = cv2.resize(template, (gray_block.shape[1], gray_block.shape[0]))
                    # Aplicar correspondência de template
                    res = cv2.matchTemplate(gray_block, resized_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    # Calcular a distância de cor
                    color_distance = -np.linalg.norm(avg_color_block - template_colors[label][idx])
                    # Combinar a pontuação de forma e cor
                    total_score = max_val + color_distance * 0.2  # Ajuste o peso da cor se necessário
                    if total_score > max_score_for_type:
                        max_score_for_type = total_score
                # Comparar o melhor score deste tipo com o melhor geral
                if max_score_for_type > best_score:
                    second_best_score = best_score
                    second_best_match = best_match
                    best_score = max_score_for_type
                    best_match = label
                elif max_score_for_type > second_best_score:
                    second_best_score = max_score_for_type
                    second_best_match = label
            block_labels[i, j] = best_match
            best_scores[i, j] = best_score
            second_best_matches[i, j] = second_best_match if second_best_match is not None else best_match
            second_best_scores[i, j] = second_best_score
    print("Mapa do Tabuleiro:")
    print(block_labels)
    return block_labels, best_scores, second_best_matches

# Função para verificar se há combinações de 3 ou mais após a troca
def has_match(grid):
    max_match_length = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            piece_type = grid[row, col]
            # Evitar peças já verificadas
            if piece_type == -1:
                continue
            # Checar sequência horizontal
            count = 1
            c = col + 1
            while c < grid_cols and grid[row, c] == piece_type:
                count += 1
                c += 1
            if count >= 3:
                max_match_length = max(max_match_length, count)
            # Checar sequência vertical
            count = 1
            r = row + 1
            while r < grid_rows and grid[r, col] == piece_type:
                count += 1
                r += 1
            if count >= 3:
                max_match_length = max(max_match_length, count)
    return max_match_length

# Função para simular uma troca e verificar se resulta em uma combinação válida
def simulate_swap_and_check_match(grid, pos1, pos2):
    # Criar uma cópia do grid
    temp_grid = grid.copy()
    # Realizar a troca
    temp_grid[pos1], temp_grid[pos2] = temp_grid[pos2], temp_grid[pos1]
    # Verificar o tamanho máximo de combinação após a troca
    match_length = has_match(temp_grid)
    return match_length

# Função para encontrar a melhor jogada considerando todas as possíveis trocas
def find_best_move(block_grid, invalid_moves):
    possible_moves = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            current_pos = (row, col)

            # Verificar troca para a direita
            if col + 1 < grid_cols:
                right_pos = (row, col + 1)
                move = (current_pos, right_pos)
                if move in invalid_moves:
                    continue
                match_length = simulate_swap_and_check_match(block_grid, current_pos, right_pos)
                if match_length >= 3:
                    possible_moves.append({
                        'from': current_pos,
                        'to': right_pos,
                        'match_length': match_length
                    })

            # Verificar troca para baixo
            if row + 1 < grid_rows:
                down_pos = (row + 1, col)
                move = (current_pos, down_pos)
                if move in invalid_moves:
                    continue
                match_length = simulate_swap_and_check_match(block_grid, current_pos, down_pos)
                if match_length >= 3:
                    possible_moves.append({
                        'from': current_pos,
                        'to': down_pos,
                        'match_length': match_length
                    })

    # Ordenar as jogadas possíveis pelo tamanho da combinação (maior primeiro)
    possible_moves.sort(key=lambda x: (-x['match_length'], -x['from'][0], x['from'][1]))

    if possible_moves:
        best_move = possible_moves[0]
        print(f"Movimento selecionado: {best_move}")
        return (best_move['from'], best_move['to'])
    else:
        return None

# Função para ajustar a identificação em áreas com possíveis erros
def adjust_identification(block_labels, best_scores, second_best_matches):
    adjusted = False
    # Verificar se há combinações existentes no tabuleiro (o que não deveria ocorrer)
    match_length = has_match(block_labels)
    if match_length >= 3:
        print("Combinação existente detectada no tabuleiro. Ajustando identificação.")
        for row in range(grid_rows):
            for col in range(grid_cols):
                piece_type = block_labels[row, col]
                # Checar sequência horizontal
                count = 1
                positions = [(row, col)]
                c = col + 1
                while c < grid_cols and block_labels[row, c] == piece_type:
                    count += 1
                    positions.append((row, c))
                    c += 1
                if count >= 3:
                    for pos in positions:
                        r, c = pos
                        block_labels[r, c] = second_best_matches[r, c]
                    adjusted = True
                # Checar sequência vertical
                count = 1
                positions = [(row, col)]
                r = row + 1
                while r < grid_rows and block_labels[r, col] == piece_type:
                    count += 1
                    positions.append((r, col))
                    r += 1
                if count >= 3:
                    for pos in positions:
                        r, c = pos
                        block_labels[r, c] = second_best_matches[r, c]
                    adjusted = True
    return adjusted

# Função para realizar a jogada com pyautogui
def execute_move(move, region, offset=mouse_offset):
    if move is None:
        print("Nenhuma jogada válida encontrada.")
        return False  # Retorna False se não executou nenhuma jogada

    start, end = move
    block_width = region["width"] // grid_cols
    block_height = region["height"] // grid_rows
    x1 = region["left"] + start[1] * block_width + block_width // 2
    y1 = region["top"] + start[0] * block_height + block_height // 2
    x2 = region["left"] + end[1] * block_width + block_width // 2
    y2 = region["top"] + end[0] * block_height + block_height // 2

    # Aplica o offset de 4 pixels na direção do movimento
    if x2 > x1:
        x2 += offset
    elif x2 < x1:
        x2 -= offset
    if y2 > y1:
        y2 += offset
    elif y2 < y1:
        y2 -= offset

    # Realiza o movimento usando pyautogui
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x2, y2, duration=0.2)
    print(f"Movimento executado de ({start[0]}, {start[1]}) para ({end[0]}, {end[1]})")
    return True  # Retorna True se executou uma jogada

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Modo Debug para Identificação de Blocos")
parser.add_argument("--debug", action="store_true", help="Mostra a identificação de cada peça")
parser.add_argument("--formato", action="store_true", help="Mostra os kernels (templates) das peças")
args = parser.parse_args()

# Carregar os templates e as médias de cores
templates, template_colors = load_templates()

# Função para exibir os kernels (templates)
def display_templates(templates):
    num_templates = sum(len(tmpl_list) for tmpl_list in templates.values())
    fig, axes = plt.subplots(1, num_templates, figsize=(num_templates * 2, 2))
    idx = 0
    for label, tmpl_list in templates.items():
        for tmpl in tmpl_list:
            axes[idx].imshow(tmpl, cmap='gray')
            axes[idx].set_title(f"Tipo {label}")
            axes[idx].axis('off')
            idx += 1
    plt.show()

if args.formato:
    # Modo para exibir os kernels (templates)
    display_templates(templates)
elif args.debug:
    # Modo de Debug: Captura e mostra identificação dos blocos sem jogar automaticamente
    try:
        while True:
            # Captura o tabuleiro
            board_image = capture_board(screen_region)
            block_labels, _, _ = process_image(board_image, templates, template_colors)

            # Desenhar o grid com IDs dos blocos
            img_with_ids = capture_and_draw_grid_with_ids(screen_region, block_labels)

            # Exibir a imagem com as identificações
            plt.imshow(img_with_ids)
            plt.axis("off")
            plt.pause(0.5)  # Pausa para atualização da visualização

            plt.clf()  # Limpa a figura para evitar sobreposição

    except KeyboardInterrupt:
        print("Modo Debug interrompido pelo usuário.")
else:
    # Modo automático para jogar
    try:
        invalid_moves = set()  # Para armazenar jogadas inválidas já tentadas
        while True:
            # Captura o tabuleiro
            board_image = capture_board(screen_region)
            block_labels, best_scores, second_best_matches = process_image(board_image, templates, template_colors)

            # Ajustar identificação se houver combinações existentes
            adjusted = adjust_identification(block_labels, best_scores, second_best_matches)
            if adjusted:
                print("Identificação ajustada. Recalculando melhor jogada.")
                # Recalcular a melhor jogada após o ajuste
                best_move = find_best_move(block_labels, invalid_moves)
            else:
                # Encontra a melhor jogada
                best_move = find_best_move(block_labels, invalid_moves)

            # Anuncia a jogada
            if best_move:
                print(f"Jogada anunciada: mover de {best_move[0]} para {best_move[1]}")
                # Executa a jogada
                move_executed = execute_move(best_move, screen_region)
                if move_executed:
                    time.sleep(move_delay)  # Aguarda para permitir que as peças se posicionem
                    # Verifica se o tabuleiro mudou após a jogada
                    new_board_image = capture_board(screen_region)
                    new_block_labels, _, _ = process_image(new_board_image, templates, template_colors)
                    if np.array_equal(block_labels, new_block_labels):
                        # A jogada não resultou em mudança; adicionar ao conjunto de jogadas inválidas
                        print("A jogada não resultou em mudança. Adicionando à lista de jogadas inválidas.")
                        invalid_moves.add(best_move)
                    else:
                        # A jogada foi bem-sucedida; limpar jogadas inválidas
                        invalid_moves.clear()
                else:
                    print("Falha ao executar a jogada.")
            else:
                print("Nenhuma jogada disponível.")
                time.sleep(move_delay)

    except KeyboardInterrupt:
        print("Jogo automático interrompido pelo usuário.")
