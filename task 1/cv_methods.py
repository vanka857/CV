from PIL import Image
import numpy as np

class Color:
    RED = 0
    GREEN = 1
    BLUE = 2

def windowed(array, window):
    return array[window[0]:window[1], window[2]:window[3]]

def load_image(path):
    return Image.open(path)

def save_image(image, path):
    image.save(path)

def image_to_array(image):
    return np.array(image)

def array_to_image(array):
    return Image.fromarray(np.array(array, dtype=np.uint8))

def vng_naive(stacked_value):

    def correct_value(x):
        x = max(0, x)
        x = min(x, 255)
        return int(x)
    
    [color, color_above, 
    NNWW, NNW, NN, NNE, NNEE, 
      WWN,  NW,  N,  NE, EEN, 
       WW,   W,  v,   E,  EE,
      WWS,  SW,  S,  SE, EES,
     SSWW, SSW, SS, SSE, SSEE] = stacked_value

    # calc grads
    G_N = abs(N - S) + abs(NN - v) + abs(NW - SW)/2 + abs(NE - SE)/2 + abs(NNW - W)/2 + abs(NNE - E)/2
    G_E = abs(E - W) + abs(EE - v) + abs(NE - NW)/2 + abs(SE - SW)/2 + abs(EEN - N)/2 + abs(EES - S)/2
    G_S = abs(S - N) + abs(SS - v) + abs(SW - NW)/2 + abs(SE - NE)/2 + abs(SSW - W)/2 + abs(SSE - E)/2
    G_W = abs(W - E) + abs(WW - v) + abs(NW - NE)/2 + abs(SW - SE)/2 + abs(WWN - N)/2 + abs(WWS - S)/2
    if color == Color.GREEN:
        G_NE = abs(NE - SW) + abs(NNEE - v) + abs(NNE - W) + abs(EEN - S)
        G_SE = abs(SE - NW) + abs(SSEE - v) + abs(SSE - W) + abs(EES - N)
        G_NW = abs(NW - SE) + abs(NNWW - v) + abs(NNW - E) + abs(WWN - S)
        G_SW = abs(SW - NE) + abs(SSWW - v) + abs(SSW - E) + abs(WWS - N)
    else:
        G_NE = abs(NE - SW) + abs(NNEE - v) + abs(NNE - N)/2 + abs(EEN - E)/2 + abs(E - S)/2 + abs(N - W)/2
        G_SE = abs(SE - NW) + abs(SSEE - v) + abs(SSE - S)/2 + abs(EES - E)/2 + abs(E - N)/2 + abs(S - W)/2
        G_NW = abs(NW - SE) + abs(NNWW - v) + abs(NNW - N)/2 + abs(WWN - W)/2 + abs(W - S)/2 + abs(N - E)/2
        G_SW = abs(SW - NE) + abs(SSWW - v) + abs(SSW - S)/2 + abs(WWS - W)/2 + abs(W - N)/2 + abs(S - E)/2
        
    # T = 1.5 * Min + 0.5 (Max + Min) = 2 * Min + 0.5 Max
    grads = np.array([G_N, G_E, G_S, G_W, G_NE, G_SE, G_NW, G_SW])
    T = 2 * np.min(grads) + 0.5 * np.max(grads)
    if T < 0.0001:
        return [v, v, v]
    
    grads_mask = grads < T
    n_grads = np.sum(grads_mask)

    N_E_S_W_NE_SE_NW_SW_0 = np.array([NN, EE, SS, WW, NNEE, SSEE, NNWW, SSWW]) + v
    NE_SE_NW_SW_1 = np.array([N, S, N, S]) + np.array([EEN, EES, WWN, WWS])
    NE_SE_NW_SW_2 = np.array([E, E, W, W]) + np.array([NNE, SSE, NNW, SSW])
    
    N_E_S_W_NE_SE_NW_SW = np.zeros((3, 8))

    if color == Color.GREEN:
        # рассматриваем ситуацию, как будто color_above == Color.RED
        # если это не так, то меняем местами RED и BLUE
        N_E_S_W_NE_SE_NW_SW[Color.GREEN] = N_E_S_W_NE_SE_NW_SW_0 / 2.
        
        N_E_S_W_NE_SE_NW_SW[Color.RED][0: 4] = np.array([N, (N + EEN + EES + S) / 4., S, (N + WWN + WWS + S) / 4.])
        N_E_S_W_NE_SE_NW_SW[Color.RED][4:  ] = NE_SE_NW_SW_1 / 2.

        N_E_S_W_NE_SE_NW_SW[Color.BLUE][0: 4] = np.array([(NNW + NNE + W + E) / 4., E, (SSW + SSE + W + E) / 4., W])
        N_E_S_W_NE_SE_NW_SW[Color.BLUE][4:  ] = NE_SE_NW_SW_2 / 2.
    
        RGB_sum = np.sum(N_E_S_W_NE_SE_NW_SW * grads_mask, axis=1)

        g = v
        b = v + (RGB_sum[Color.BLUE] - RGB_sum[Color.GREEN]) / n_grads
        r = v + (RGB_sum[Color.RED] - RGB_sum[Color.GREEN]) / n_grads

        if color_above != Color.RED:
            temp = r
            r = b
            b = temp

    else:
        N_E_S_W_NE_SE_NW_SW[Color.GREEN][0: 4] = np.array([N, E, S, W])
        N_E_S_W_NE_SE_NW_SW[Color.GREEN][4:  ] = (NE_SE_NW_SW_1 + NE_SE_NW_SW_2) / 4.

        N_E_S_W_NE_SE_NW_SW[Color.RED] = N_E_S_W_NE_SE_NW_SW_0 / 2.

        N_E_S_W_NE_SE_NW_SW[Color.BLUE][0: 4] = (np.array([NE, NE, SW, NW]) + np.array([NW, SE, SE, SW])) / 2.
        N_E_S_W_NE_SE_NW_SW[Color.BLUE][4:  ] = np.array([NE, SE, NW, SW])

        RGB_sum = np.sum(N_E_S_W_NE_SE_NW_SW * grads_mask, axis=1)

        b = v
        g = v + (RGB_sum[Color.GREEN] - RGB_sum[Color.RED]) / n_grads
        r = v + (RGB_sum[Color.BLUE] - RGB_sum[Color.RED]) / n_grads
        
        if color == Color.RED:
            temp = r
            r = b
            b = temp
            
        g = correct_value(g)
        b = correct_value(b)
        r = correct_value(r)
        
    return [r, g, b]

def process_image_naive(cfa_array, window=None):
    heigth, width = cfa_array.shape

    # Создадим фильтр Байера
    pattern = [[Color.RED, Color.GREEN],
               [Color.GREEN, Color.BLUE]]

    bayer_filter = np.tile(pattern, (heigth // 2 + 1, width // 2 + 1))[:heigth, :width]
    
    # фильтр Байера для строки над текущей (пригодится при реализации алгоритма)
    temp = np.pad(bayer_filter, 1)
    bayer_filter_above = temp[0:-2, 1:-1]

    # Создадим смещения (клетку 5х5)
    slide_view = np.lib.stride_tricks.sliding_window_view(np.pad(cfa_array, 2), (5, 5))

    t = np.swapaxes(np.swapaxes(slide_view, 0, 2), 1, 3)
    t = t.reshape(-1, t.shape[2], t.shape[3])

    [NNWW, NNW, NN, NNE, NNEE, 
      WWN,  NW,  N,  NE, EEN , 
       WW,   W,  v,   E,  EE ,
      WWS,  SW,  S,  SE, EES ,
     SSWW, SSW, SS, SSE, SSEE] = t

    # упакуем всё в матрицу размером, соответствующим размеру входного изображения
    stacked_matrix = np.stack((bayer_filter, bayer_filter_above, 
    NNWW, NNW, NN, NNE, NNEE, 
      WWN,  NW,  N,  NE, EEN, 
       WW,   W,  v,   E,  EE,
      WWS,  SW,  S,  SE, EES,
     SSWW, SSW, SS, SSE, SSEE), axis=-1)

    # window -- часть картинки, к которой применяем vng
    res = None
    if window is None:
        res = np.apply_along_axis(vng_naive, -1, stacked_matrix)
    else:
        res = np.apply_along_axis(vng_naive, -1, windowed(stacked_matrix, window))

    res = 255 - res
    return res
