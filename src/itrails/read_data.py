import numpy as np
from Bio import AlignIO
from numba import njit


@njit
def get_obs_state_dct():
    """Returns a list of all possible 4-character nucleotide state strings based on 'A', 'C', 'T', 'G' and, if not already present, appends additional states using 'N'.

    :return: list of observed state strings.
    :rtype: list[str]."""
    lst = []
    for a in ["A", "C", "T", "G"]:
        for b in ["A", "C", "T", "G"]:
            for c in ["A", "C", "T", "G"]:
                for d in ["A", "C", "T", "G"]:
                    lst.append(a + b + c + d)
    for a in ["A", "C", "T", "G", "N"]:
        for b in ["A", "C", "T", "G", "N"]:
            for c in ["A", "C", "T", "G", "N"]:
                for d in ["A", "C", "T", "G", "N"]:
                    if (a + b + c + d) not in lst:
                        lst.append(a + b + c + d)
    return lst


@njit
def get_obs_state_dct_new_method():
    """Returns a list of observed state strings using a new method that generates 3-character nucleotide strings from 'A', 'C', 'T', 'G' and appends additional states using 'N' if not already present.

    :return: list of observed state strings using the new method.
    :rtype: list[str]."""
    lst = []
    for a in ["A", "C", "T", "G"]:
        for b in ["A", "C", "T", "G"]:
            for d in ["A", "C", "T", "G"]:
                lst.append(a + b + d)
    for a in ["A", "C", "T", "G", "N"]:
        for b in ["A", "C", "T", "G", "N"]:
            for d in ["A", "C", "T", "G", "N"]:
                if (a + b + d) not in lst:
                    lst.append(a + b + d)
    return lst


@njit
def get_idx_state(state):
    """Given a state index, returns an array of resolved state indices by recursively replacing any ambiguous 'N' in the observed state string with 'A', 'C', 'T', and 'G'.

    :param state: index of the observed state in the state dictionary.
    :type state: int.
    :return: numpy array of resolved state indices.
    :rtype: np.ndarray."""
    lst = get_obs_state_dct()
    st = lst[state]
    idx = st.find("N")
    if idx == -1:
        return np.array([state])
    else:
        return np.concatenate(
            (
                get_idx_state(lst.index(st[:idx] + "A" + st[idx + 1 :])),
                get_idx_state(lst.index(st[:idx] + "C" + st[idx + 1 :])),
                get_idx_state(lst.index(st[:idx] + "T" + st[idx + 1 :])),
                get_idx_state(lst.index(st[:idx] + "G" + st[idx + 1 :])),
            )
        )


@njit
def get_idx_state_new_method(state):
    """Given a state index using the new observed state dictionary, returns an array of resolved state indices by recursively replacing any ambiguous 'N' in the observed state string with 'A', 'C', 'T', and 'G'.

    :param state: index of the observed state in the new state dictionary.
    :type state: int.
    :return: numpy array of resolved state indices.
    :rtype: np.ndarray."""
    lst = get_obs_state_dct_new_method()
    st = lst[state]
    idx = st.find("N")
    if idx == -1:
        return np.array([state])
    else:
        return np.concatenate(
            (
                get_idx_state(lst.index(st[:idx] + "A" + st[idx + 1 :])),
                get_idx_state(lst.index(st[:idx] + "C" + st[idx + 1 :])),
                get_idx_state(lst.index(st[:idx] + "T" + st[idx + 1 :])),
                get_idx_state(lst.index(st[:idx] + "G" + st[idx + 1 :])),
            )
        )


def maf_parser(file, sp_lst):
    """Parses a MAF file to extract sequence alignments for the specified species. for each alignment block, collects sequences for species in sp_lst, replaces gaps '-' with 'N', and converts each column of nucleotides to an index using the observed state dictionary from get_obs_state_dct.

    :param file: path to the MAF file. :type file: str.
    :param sp_lst: list of species names (expected length 4) to extract sequences for.
    :type sp_lst: list[str].
    :return: list of numpy arrays where each array contains the state indices for a block.
    :rtype: list[np.ndarray]."""
    order_st = get_obs_state_dct()
    total_lst = []
    loglik_acc = 0
    for multiple_alignment in AlignIO.parse(file, "maf"):
        dct = {}
        for seqrec in multiple_alignment:
            if seqrec.name.split(".")[0] in sp_lst:
                dct[seqrec.name.split(".")[0]] = str(seqrec.seq).replace("-", "N")
        if len(dct) == 4:
            idx_lst = np.zeros((len(seqrec.seq)), dtype=np.int64)
            for i in range(len(seqrec.seq)):
                idx_lst[i] = order_st.index(
                    "".join([dct[j][i] for j in sp_lst]).upper()
                )
            total_lst.append(idx_lst)
    return total_lst


def maf_parser_new_method(file, sp_lst):
    """Parses a MAF file to extract sequence alignments for the specified species using the new observed state dictionary. for each alignment block, collects sequences for species in sp_lst, replaces gaps '-' with 'N', and converts each column of nucleotides to an index using the observed state dictionary from get_obs_state_dct_new_method.

    :param file: path to the MAF file. :type file: str.
    :param sp_lst: list of species names (expected length 4) to extract sequences for.
    :type sp_lst: list[str].
    :return: list of numpy arrays where each array contains the state indices for a block. :rtype: list[np.ndarray].
    """
    order_st = get_obs_state_dct_new_method()
    total_lst = []
    loglik_acc = 0
    for multiple_alignment in AlignIO.parse(file, "maf"):
        dct = {}
        for seqrec in multiple_alignment:
            if seqrec.name.split(".")[0] in sp_lst:
                dct[seqrec.name.split(".")[0]] = str(seqrec.seq).replace("-", "N")
        if len(dct) == 4:
            idx_lst = np.zeros((len(seqrec.seq)), dtype=np.int64)
            for i in range(len(seqrec.seq)):
                idx_lst[i] = order_st.index(
                    "".join([dct[j][i] for j in sp_lst]).upper()
                )
            total_lst.append(idx_lst)
    return total_lst


def parse_coordinates(file, sp_lst, ref):
    """Parses the coordinates of a MAF file polarized by a reference sequence. The reference does not need to be one of the species used for Viterbi or posterior decoding.

    :param file: path to the MAF file. 
    :type file: str.
    :param sp_lst: list of species names (expected length 4) to extract sequences for.
    :type sp_lst: list[str].
    :param ref: name of the reference to polarize the coordinates.
    :type ref: str.
    :return: list of numpy arrays where each array contains the state indices for a block. :rtype: list[list].
    """
    seq_list = []
    start_list = []
    strand_list = []
    chromsize_list = []
    # For each block
    for multiple_alignment in AlignIO.parse(file, "maf"):
        start = float('nan')
        ref_seq = ''
        # For each sequence in the alignment block
        acc = 0
        for seqrec in multiple_alignment:
            if seqrec.name.split('.')[0] in sp_lst:
                length = len(str(seqrec.seq))
                acc += 1
            if seqrec.name.split('.')[0] == ref:
                # Save the start position of the block in human coordinates
                start = seqrec.annotations['start']
                # Save nucleotides as ones and gaps as zeros
                ref_seq = ''.join([str(int(j!='-')) for j in str(seqrec.seq)])
                strand = seqrec.annotations['strand']
                chrom_size = seqrec.annotations["srcSize"]
                
        # Append information to list
        if acc == 4: 
            if ref_seq != '':
                start_list.append(start)
                seq_list.append(ref_seq)
                strand_list.append(strand)
                chromsize_list.append(chrom_size)
            else:
                start_list.append(start)
                seq_list.append('0'*length)
                strand_list.append(0)
                chromsize_list.append(0)

    tot_lst = []
    # For each block
    for i in range(len(start_list)):
        # If forward strand, save coordinate as is
        if strand_list[i] == 1:
            st = start_list[i]
        # If reverse strand, save coordinate in reverse
        elif strand_list[i] == -1:
            st = chromsize_list[i] - start_list[i]
        # If sequence is missing, append -9
        elif strand_list[i] == 0:
            tot_lst.append([-9]*len(seq_list[i]))
            continue
        tmp_lst = []
        # For each site in the ref sequence
        for j in seq_list[i]:
            # If the site is not a gap
            if j == '1':
                # Append position
                tmp_lst.append(st)
                # Update coordinate (backwards if strand is -1)
                st += strand_list[i]
            # If it is a gap
            else:
                # Append -9
                tmp_lst.append(-9)
        tot_lst.append(tmp_lst)
    
    return tot_lst

