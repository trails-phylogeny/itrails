import numpy as np
from trans_mat import wrapper_state, get_trans_mat
from expm import expm
from cut_times import cutpoints_AB, cutpoints_ABC, get_times
from combine_states import combine_states
from numba.typed import Dict
from numba.types import Tuple, int64
from numba import njit
import time

int_tuple = Tuple((int64, int64))


@njit
def get_omegas_numba(omega_nums):
    omega_dict = Dict.empty(
        key_type=int_tuple,
        value_type=int_tuple,
    )

    for i in range(len(omega_nums)):
        for j in range(len(omega_nums)):
            omega_dict[(omega_nums[i], omega_nums[j])] = (omega_nums[i], omega_nums[j])

    return omega_dict


@njit
def get_final_per_interval_nodict(trans_mat_ab, times, omega_dict, pi_AB):
    # Create dictionary for base case, also every omega will be called -1, -1.
    time_0dict = {}
    all_omegas = (-1, -1)
    mat_0 = np.zeros_like(trans_mat_ab)
    # Base case for time 0, from all omegas to every other possibility.
    exponential_time_0 = expm(trans_mat_ab * times[0])
    time0 = time.time()
    for key, value in omega_dict.items():
        exponential_time_add = mat_0.copy()  # 15x15 0s
        exponential_time_add[:, min(value) : max(value) + 1] = exponential_time_0[
            :, min(value) : max(value) + 1
        ]
        time_0dict[(all_omegas, key)] = exponential_time_add
    # Dictionary to accumulate the results, keys are omega paths as tuples, values are precomputed matrices.
    acc_results = time_0dict

    # Each of the time cuts
    for i in range(1, len(times)):
        exponential_time = expm(trans_mat_ab * times[i])
        each_time_dict = {}
        actual_results = {}
        # Populate a temp dictionary with the every possible slice.
        for key, value in omega_dict.items():
            for key2, value2 in omega_dict.items():
                if key[0] <= key2[0] and key[1] <= key2[1]:
                    exponential_time_add = mat_0.copy()
                    exponential_time_add[
                        min(value) : max(value) + 1, min(value2) : max(value2) + 1
                    ] = exponential_time[
                        min(value) : max(value) + 1, min(value2) : max(value2) + 1
                    ]

                    each_time_dict[(key, key2)] = exponential_time_add
        # Multiply each possible slice with the results that we already had and update the accumulated results dictionary.
        for transition_0, matrix_0 in acc_results.items():
            end_state = transition_0[-1]
            for transition_1, matrix_1 in each_time_dict.items():
                start_state = transition_1[0]
                if start_state == end_state:
                    result = matrix_0 @ matrix_1
                    actual_results[(*transition_0, transition_1[1])] = result
                else:
                    continue
        acc_results = actual_results

    # Create the final_p vector multiplying each slice by pi and populate it with the calculation for each state we end up in.
    # Final prob vector debug is just a sum of every to see if they sum up to 1
    final_prob_vector_debug = np.zeros((trans_mat_ab.shape[0]), dtype=np.float64)
    final_prob_vector = {}
    for trans, prob_slice in acc_results.items():
        pi_slice = pi_AB @ prob_slice
        final_prob_vector[trans] = pi_slice
        for i in range(len(final_prob_vector_debug)):
            final_prob_vector_debug[i] += pi_slice[i]

    time1 = time.time()
    print(time1 - time0)
    return final_prob_vector_debug, acc_results, final_prob_vector


def get_final_per_interval(trans_mat_ab, times, omega_dict, pi_AB):

    # Create dictionary for base case, also every omega will be called -1, -1.
    time_0dict = {}
    all_omegas = (-1, -1)
    mat_0 = np.zeros_like(trans_mat_ab)
    # Base case for time 0, from all omegas to every other possibility.
    exponential_time_0 = expm(trans_mat_ab * times[0])
    time0 = time.time()
    for key, value in omega_dict.items():
        exponential_time_add = mat_0.copy()  # 15x15 0s
        exponential_time_add[:, min(value) : max(value) + 1] = exponential_time_0[
            :, min(value) : max(value) + 1
        ]
        time_0dict[(all_omegas, key)] = exponential_time_add
    # Dictionary to accumulate the results, keys are omega paths as tuples, values are precomputed matrices.
    acc_results = time_0dict

    # acc: keys: -1-1, 00///-1-1, 03///-1-1, 30///-1-1, 33
    # acc: values: matrices (slices)

    # Each of the time cuts
    for i in range(1, len(times)):
        exponential_time = expm(trans_mat_ab * times[i])
        each_time_dict = {}
        actual_results = {}
        # Populate a temp dictionary with the every possible slice.
        for key, value in omega_dict.items():
            for key2, value2 in omega_dict.items():
                if (
                    key[0] <= key2[0] and key[1] <= key2[1]
                ):  # Cuidado para 3, de 3 a 5 no puede pasar. ######
                    exponential_time_add = mat_0.copy()
                    exponential_time_add[
                        min(value) : max(value) + 1, min(value2) : max(value2) + 1
                    ] = exponential_time[
                        min(value) : max(value) + 1, min(value2) : max(value2) + 1
                    ]

                    each_time_dict[(key, key2)] = (
                        exponential_time_add  # Initialize each time dict con matrices de 0s #############
                    )
                    # Generarla entera para todos los times, sin tener que mantener matrices en memoria y "mask" con 0s

        # No tener que guardar todas las matrices del primer nested forloop. Podemos utilizar todos los paths the omegas
        # Podemos sacar todas las omegas y despues on the fly sacamos todos los caminos "de golpe"

        # Multiply each possible slice with the results that we already had and update the accumulated results dictionary.
        for transition_0, matrix_0 in acc_results.items():
            end_state = transition_0[-1]
            for transition_1, matrix_1 in each_time_dict.items():
                start_state = transition_1[0]
                if start_state == end_state:
                    result = matrix_0 @ matrix_1  # Multiplicar por omega vector
                    actual_results[(*transition_0, transition_1[1])] = result
                else:
                    continue
        acc_results = actual_results

    # Create the final_p vector multiplying each slice by pi and populate it with the calculation for each state we end up in.
    # Final prob vector debug is just a sum of every to see if they sum up to 1
    final_prob_vector_debug = np.zeros((trans_mat_ab.shape[0]), dtype=np.float64)
    final_prob_vector = {}
    for trans, prob_slice in acc_results.items():
        pi_slice = pi_AB @ prob_slice
        final_prob_vector[trans] = pi_slice
        for i in range(len(final_prob_vector_debug)):
            final_prob_vector_debug[i] += pi_slice[i]

    time1 = time.time()
    print(time1 - time0)
    return final_prob_vector_debug, acc_results, final_prob_vector


def get_joint_prob_mat(
    t_A,
    t_B,
    t_AB,
    t_C,
    rho_A,
    rho_B,
    rho_AB,
    rho_C,
    rho_ABC,
    coal_A,
    coal_B,
    coal_AB,
    coal_C,
    coal_ABC,
    n_int_AB,
    # n_int_ABC,
    p_init_A=np.array([0, 1], dtype=np.float64),
    p_init_B=np.array([0, 1], dtype=np.float64),
    p_init_C=np.array([0, 1], dtype=np.float64),
    cut_AB="standard",
    # cut_ABC="standard",
    # tmp_path="./",
):

    ## This wouold be faster reading from csv but I think generating it from scratch is more mantainable
    # State space 1 seq
    transitions_1, omega_dict_1, state_dict_1 = wrapper_state(1)
    # State space 2 seq
    transitions_2, omega_dict_2, state_dict_2 = wrapper_state(2)
    # State space 3 seq
    transitions_3, omega_dict_3, state_dict_3 = wrapper_state(3)
    print(omega_dict_2)
    # Get transition matrix for A
    trans_mat_a = get_trans_mat(transitions_1, 1, coal_A, rho_A)
    # Get transition matrix for B
    trans_mat_b = get_trans_mat(transitions_1, 1, coal_B, rho_B)
    # Get transition matrix for C
    trans_mat_c = get_trans_mat(transitions_1, 1, coal_C, rho_C)
    # Get transition matrix for AB
    trans_mat_ab = get_trans_mat(transitions_2, 2, coal_AB, rho_AB)
    # Get transition matrix for ABC
    trans_mat_abc = get_trans_mat(transitions_3, 3, coal_ABC, rho_ABC)

    # Get final probs A
    final_A = p_init_A @ expm(trans_mat_a * t_A)
    # Get final probs B
    final_B = p_init_B @ expm(trans_mat_b * t_B)
    # Get final probs C
    final_C = p_init_C @ expm(trans_mat_c * t_C)

    # Dicts to know position of hidden states in the prob matrix.
    number_dict_A = state_dict_1
    number_dict_B = state_dict_1
    number_dict_C = state_dict_1
    number_dict_AB = state_dict_2
    number_dict_ABC = state_dict_3

    # Combine A and B CTMCs
    pi_AB = combine_states(
        number_dict_A, number_dict_B, number_dict_AB, final_A, final_B
    )
    # Get cutopints and times based on cutopoints.
    if isinstance(cut_AB, str):
        cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    times_AB = get_times(cut_AB, list(range(len(cut_AB))))

    """ 
    final_AB pi_AB por cada vector final del path, luego combine states me daria
     """
    # Calculate the final probability vector for the two sequence CTMC
    get_final_per_interval(trans_mat_ab, times_AB, omega_dict_2, pi_AB)

    # Combine AB and C to get initial probabilities for the three sequence CTMC
    # pi_ABC = combine_states(
    #    number_dict_AB, number_dict_C, number_dict_ABC, final_AB, final_C
    # )

    pass


get_joint_prob_mat(
    t_A=10,
    t_B=10,
    t_AB=20,
    t_C=20,
    rho_A=0.3,
    rho_B=0.4,
    rho_AB=0.6,
    rho_C=0.3,
    rho_ABC=0.4,
    coal_A=0.6,
    coal_B=0.4,
    coal_AB=0.2,
    coal_C=0.5,
    coal_ABC=0.4,
    n_int_AB=3,
    p_init_A=np.array([0, 1], dtype=np.float64),
    p_init_B=np.array([0, 1], dtype=np.float64),
    p_init_C=np.array([0, 1], dtype=np.float64),
    cut_AB="standard",
)


""" 
0. Probar con 0s - HECHO, 1 orden de magnitud + rápido

0. GitHub - HECHO, invitación a Iker

1. njitear - NO, sigo usando dictionaries, pensar forma de cambiarlo

2. modificar function para que returnee las cosas separadas - NO/HECHO, cambiar implementación de combine_states

3. Se puede hacer independiente y luego juntar sumarlos todos y tenener init vec, combinar
con final_C y con eso start prob de 3 seq). Podemos generar todos los paths sin tener en cuenta lo del principio.

4. La info del omega que empieza esta en pi_ABC.

Siguiente paso mismo concepto pero condicionar en cutpoints en los posibles paths
Despues seria matchear 2 spec and 3 spec. Si cogemos un path especifico en la join tprob matrix saber donde esta.

Hay poca info en 1 seq para ne. En trails original igualamos las ne.

Mantemeos variable y cuando hagamos la optimizacion ponemos que estandar sea la misma ne.

--IDEAS DURANTE LA SEMANA

Utilzar 0s ha hecho la implementacion 1 orden de magnitud más rapida.

Github organization hecha, invitar a Iker, antes de crear el repo quiero tener más o menos limpio este código, mid June código limpio.

Quiero dedicar tiempo a limpiar lo que tengo también, puedo quitar muchos diccionarios ahora
que usamos 0 porque ya esta implicit que omegas pueden ser y cuales no en los 0 del vector.

Pensar como pasarlo al 3 sequence, esta semana y hasta el 7 de junio no voy a tener demasiado tiempo por examenes.

--REUNION 28/5
3D numpy array.
GPU?
Estructura de package
Modules



Ideas Post-Examenes
- Hay cosas que solo se tienen que callear una vez como por ejemplo los numeritos de omega
etc, se podría dar como args a la función general y que todo lo de la general pueda estar
jitted porque hay mucho forloop.

- Numba permite usar typed dict, implementado:
    - Los omegas con numba en typed dict
    - Casi todo el array de numeros etc con numba salvo lexsort, buscar como
    - GPU: Hay una forma de speed up matrix multiplication using numba+CUDA (https://nyu-cds.github.io/python-numba/05-cuda/)

    
12/6/24
    - Preallocate en omegas
    - Benchmark matmul CUDA

Week 25
    - Benchmark matmul CUDA-más lento, creo que puede ser porque
    al pasar a grafica y sacar de grafica se tarda tiempo y no se
    compensa con la mejora en velocidad sobre el @

    - Implementados los omegas con binary encoding de un vector
    con np.where se sacan los indices para la matrix


    - Binary en vez de int64.
    - En number array no hace falta forzar el orden.
    - No hace falta guardar cada estado en number_array, solo los omegas.
 """
