"""
Module to define configuration and styles
"""

# Classes
#_______________________________________________________________________________
class Colors:
    """
    Set the format for coloring text on the screen
    """
    RESET = "\033[0m"
    NEGRITA = "\033[1m"
    SUBRAYADO = "\033[4m"
    ROJO = "\033[91m"
    VERDE = "\033[92m"
    AMARILLO = "\033[93m"
    AZUL = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    CYAN_2 = "\033[36m"
    BLANCO = "\033[30m"
    NEGRO = "\033[37m"


# Variables
#_______________________________________________________________________________

# Real name of the diferent variables of the BioFuel datasets
bio_fuel_vars = {
    'var_x1': 'LHSV',
    'var_x2': 'P_H2',
    'var_x3': 'P_H2S',
    'var_x4': 'S_carga',
    'var_x5': 'N_carga',
    'var_x6': 'rho_0',
    'var_x7': 'D86T90_carga',
    'var_x8': 'S_prod',
    'var_x9': 'Kcalib',
    'var_y': 'WABT'
}

# Domain of the BioFuel datasets values
biofuel_range = {
    'var_x1': (0.5, 1),
    'var_x2': (24, 40),
    'var_x3': (0.008, 0.012),
    'var_x4': (5000, 45000),
    'var_x5': (100, 500),
    'var_x6': (0.82, 0.875),
    'var_x7': (325, 365),
    'var_x8': (0.1, 30),
    'var_x9': (0.1, 1.2),
    'Kcalib': (0.1, 1.2),
    'var_y': (290, 425)
}