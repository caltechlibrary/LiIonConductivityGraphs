import os
import re
import pandas as pd
import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure, Species, Element, PeriodicSite, Composition, DummySpecies
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation, DiscretizeOccupanciesTransformation
from pymatgen.core import periodic_table as pt
from pymatgen.analysis.structure_matcher import StructureMatcher
from scipy.spatial import cKDTree
import random
from func_timeout import func_timeout, FunctionTimedOut
import copy

from ElMD import ElMD
from ElM2D import ElM2D


def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(file_path)


def load_cif_structure(icsd, cif_dir):
    """
    Load a CIF structure given an ICSD entry and a directory containing CIF files.

    Args:
        icsd (str): ICSD entry.
        cif_dir (str): Directory containing CIF files.

    Returns:
        structure: Pymatgen Structure object if CIF file exists, otherwise None.
    """
    # Construct path to CIF file
    cif_file_path = os.path.join(cif_dir, f"{icsd}.cif")

    # Check if CIF file exists
    if os.path.exists(cif_file_path):
        # Parse CIF file
        try:
            cif_parser = CifParser(cif_file_path, occupancy_tolerance=2.0)
            structure = cif_parser.get_structures()[0]
            return structure
        except Exception as e:
            print(e)
            print(f"The structure could not be loaded for ICSD entry {icsd}")
            return None
    else:
        # Print message if CIF file not found
        print(f"No cif found for ICSD entry {icsd}")
        return None


def load_ordered_cif_structure(icsd, cif_dir):
    """
    Load an ordered CIF structure given an ICSD entry and a directory containing CIF files.

    Args:
        icsd (str): ICSD entry.
        cif_dir (str): Directory containing CIF files.

    Returns:
        structure: Pymatgen Structure object if CIF file exists, otherwise None.
    """
    # Construct path to CIF file
    cif_file_path = os.path.join(cif_dir, f"{icsd}_ordered.cif")

    # Check if CIF file exists
    if os.path.exists(cif_file_path):
        # Parse CIF file
        try:
            cif_parser = CifParser(cif_file_path, occupancy_tolerance=2.0)
            structure = cif_parser.get_structures()[0]  # Assuming there is only one structure in the CIF
            return structure
        except Exception as e:
            print(e)
            print(f"The structure could not be loaded for ICSD entry {icsd}")
            return None
    else:
        # Print message if CIF file not found
        print(f"No cif found for ICSD entry {icsd}")
        return None


def remove_li_from_mixed_sites(struc):
    """
    Remove Li species from mixed sites in the structure.

    Args:
        struc (Structure): Pymatgen Structure object.

    Returns:
        Structure: Structure object with Li species removed from mixed sites.
    """
    new_sites = []

    # Iterate over each site in the structure
    for i in range(len(struc.sites)):
        # Check if a site contains only Li
        if len(struc.species_and_occu[i]) == 1 and list(struc.species_and_occu[i].keys())[0].symbol == "Li":
            # Skip this site if it contains only Li
            continue

        # Create a new species_and_occu list without Li
        new_species_and_occu = Composition(
            {sp: occu for sp, occu in struc.species_and_occu[i].items() if sp != Element("Li")})

        # Create a new site with the modified species_and_occu and coordinates
        new_site = PeriodicSite(new_species_and_occu, struc.sites[i].frac_coords, struc.lattice)
        new_sites.append(new_site)

    # Create a new structure with the modified sites
    new_struc = Structure.from_sites(new_sites)
    return new_struc



def order_structure(struc, num_return_struc=1):
    """
    Attempts to order a disordered structure using the OrderDisorderedStructureTransformation.

    Args:
        struc (Structure): Pymatgen Structure object representing the disordered structure.
        num_return_struc (int): Number of ordered structures to return in the ranked list (default is 1).

    Returns:
        list or None: A ranked list of ordered structures if successful, or None if the ordering fails.
    """
    try:
        order_transform = OrderDisorderedStructureTransformation()
        ordered_list = order_transform.apply_transformation(struc, return_ranked_list=num_return_struc)
        return ordered_list
    except Exception as e:
        return None



def supercell_factorizer(struc, factor):
    """
    Generates a supercell of a given structure based on the specified multiplication factor.

    Args:
        struc (Structure): Pymatgen Structure object representing the crystal structure.
        factor (int): The multiplication factor to use for generating the supercell. 
                      Common factors have predefined supercell configurations.

    Returns:
        Structure: The input structure scaled to form a supercell by applying the given factor.
    """
    abc_lengths = struc.lattice.lengths
    a, b, c = np.argsort(abc_lengths)
    unaltered = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    if factor in unaltered:
        supercell = [factor, 1, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    if factor == 4:
        supercell = [2, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 6:
        supercell = [3, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 8:
        supercell = [2, 2, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 9:
        supercell = [3, 3, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 10:
        supercell = [5, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 12:
        supercell = [3, 2, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 14:
        supercell = [7, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 15:
        supercell = [5, 3, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 16:
        supercell = [4, 2, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 18:
        supercell = [3, 3, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 20:
        supercell = [5, 2, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 21:
        supercell = [7, 3, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 22:
        supercell = [11, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 24:
        supercell = [2, 3, 4]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 25:
        supercell = [5, 5, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 26:
        supercell = [13, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 27:
        supercell = [3, 3, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 28:
        supercell = [7, 2, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 30:
        supercell = [5, 3, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 32:
        supercell = [4, 4, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 33:
        supercell = [11, 3, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 34:
        supercell = [17, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 35:
        supercell = [7, 5, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 36:
        supercell = [6, 3, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 38:
        supercell = [19, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 39:
        supercell = [13, 3, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 40:
        supercell = [5, 4, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 42:
        supercell = [7, 3, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 44:
        supercell = [11, 2, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 45:
        supercell = [5, 3, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 46:
        supercell = [23, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 48:
        supercell = [4, 4, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 49:
        supercell = [7, 7, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 50:
        supercell = [5, 5, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 51:
        supercell = [17, 3, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 52:
        supercell = [13, 2, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 54:
        supercell = [6, 3, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 55:
        supercell = [11, 5, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 56:
        supercell = [7, 4, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 57:
        supercell = [19, 3, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 58:
        supercell = [29, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 60:
        supercell = [5, 4, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 62:
        supercell = [31, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 63:
        supercell = [7, 3, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 64:
        supercell = [4, 4, 4]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 65:
        supercell = [13, 5, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 66:
        supercell = [11, 3, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 68:
        supercell = [17, 2, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 69:
        supercell = [23, 3, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 70:
        supercell = [7, 5, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 72:
        supercell = [6, 4, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 74:
        supercell = [37, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 75:
        supercell = [5, 5, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 76:
        supercell = [19, 2, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 77:
        supercell = [11, 7, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 78:
        supercell = [13, 3, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 80:
        supercell = [5, 4, 4]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 81:
        supercell = [9, 3, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 82:
        supercell = [41, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 84:
        supercell = [7, 4, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 85:
        supercell = [17, 5, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 86:
        supercell = [43, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 87:
        supercell = [29, 3, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 88:
        supercell = [11, 4, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 90:
        supercell = [6, 5, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 91:
        supercell = [13, 7, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 92:
        supercell = [23, 2, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 93:
        supercell = [31, 3, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 94:
        supercell = [47, 2, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 95:
        supercell = [19, 5, 1]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 96:
        supercell = [6, 4, 4]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 98:
        supercell = [7, 7, 2]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 99:
        supercell = [11, 3, 3]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    elif factor == 100:
        supercell = [5, 5, 4]
        struc.make_supercell([supercell[a], supercell[b], supercell[c]])
        return struc
    else:
        print(f"Factor missing: {factor}")


def discretize_occupancies(struc, max_denom):
    """
    Discretizes the occupancies of a disordered structure by trying different denominators.

    Args:
        struc (Structure): Pymatgen Structure object representing the disordered structure.
        max_denom (int): Maximum denominator to try for discretizing the occupancies.

    Returns:
        Structure: The structure with discretized occupancies, if successful. If no valid 
                   discretization is found, returns None.
    """
    #trans = ConventionalCellTransformation()
    #conv_struc = trans.apply_transformation(struc)
    conv_struc = struc.copy()
    discretized_struc = None

    for denom in np.arange(2, max_denom, 1):
        try:
            temp_struc = conv_struc.copy()
            trans = DiscretizeOccupanciesTransformation(max_denominator=denom, fix_denominator=True)
            discretized_struc = trans.apply_transformation(temp_struc)
            if set(discretized_struc.symbol_set) != set(struc.symbol_set):
                print("Elements missing in discretized structure!")
                raise ValueError
            for site in discretized_struc:
                if site.species_string == "":
                    print("Max denominator of {} caused missing species error".format(denom))
                    raise ValueError
            print('Discretization max denominator with fixed denominator: ', denom)
            return discretized_struc
        except:
            next

    for denom in np.arange(2, max_denom, 1):
        try:
            temp_struc = struc.copy()
            trans = DiscretizeOccupanciesTransformation(max_denominator=denom, fix_denominator=False)
            discretized_struc = trans.apply_transformation(temp_struc)
            if set(discretized_struc.symbol_set) != set(struc.symbol_set):
                print("Elements missing in discretized structure!")
                raise ValueError
            for site in discretized_struc:
                if site.species_string == "":
                    print("Max denominator of {} caused missing species error")
                    raise ValueError
            print('Discretization max denominator with unfixed denominator: ', denom)
            return discretized_struc
        except:
            next

    return discretized_struc


def supercell_helper(struc):
    """
    Attempts to order a disordered structure by applying the OrderDisorderedStructureTransformation.

    Args:
        struc (Structure): Pymatgen Structure object representing the disordered structure.

    Returns:
        list: A ranked list of ordered supercells of the structure.
    """
    trans = OrderDisorderedStructureTransformation()
    supercell_struc = trans.apply_transformation(struc, return_ranked_list=1)
    return supercell_struc


def ordering_supercell_generation(struc):
    """
    Attempts to discretize and order a disordered structure by generating supercells.

    Args:
        struc (Structure): Pymatgen Structure object representing the disordered structure.

    Returns:
        Structure: The discretized and ordered structure, if successful. 
                   If the structure remains disordered, returns the original structure.
    """
    discretized_struc = discretize_occupancies(struc, 100)
    if discretized_struc.is_ordered:
        print('Discretizing ordered the structure')
        return discretized_struc
    for multiplier in range(2, 100):
        try:
            test_struc = copy.deepcopy(discretized_struc)
            test_struc = supercell_factorizer(test_struc, multiplier)
            # this supercell_helper function attempts to order the structure
            _ = func_timeout(10, supercell_helper, kwargs={'struc': test_struc})
            # if the ordering continues for 10 s, then the timeout catches it
            # this means that the "Occupancy fractions not consistent" error was avoided
        except FunctionTimedOut:
            print('The chosen supercell multiplier was: ', multiplier)
            break
        except:
            print('Occupancy error for supercell multiplier of {}'.format(multiplier))
            continue

    return test_struc


def generate_ordered_struc_list(struc, num_return_struc=250):
    """
    Generates a list of ordered structures from a disordered input structure.

    Args:
        struc (Structure): Pymatgen Structure object representing the disordered structure.
        num_return_struc (int): The number of ordered structures to return from the transformation.

    Returns:
        list: A list of ordered configurations, ranked based on the transformation.
              Returns None if no valid supercell could be found.
    """
    order_transform = OrderDisorderedStructureTransformation()
    
    # First, attempt simple ordering
    try:
        ordered_list = order_transform.apply_transformation(struc, return_ranked_list=num_return_struc)
        return ordered_list
    except Exception as e:
        # Proceed with supercell generation if simple ordering fails
        print(f"Simple ordering failed: {e}. Attempting supercell generation...")
        temp_struc = ordering_supercell_generation(struc)
        
        if temp_struc.is_ordered:
            return [temp_struc]  # Return ordered structure in a list

        # Try ordering after supercell generation
        try:
            ordered_config_list = order_transform.apply_transformation(temp_struc, return_ranked_list=num_return_struc)
            return ordered_config_list
        except Exception as e:
            print(f"Supercell ordering failed: {e}. No valid ordered structure found.")
            return None  # Return None if everything fails


def remove_charges(struc):
    """
    Remove charges from the species in the structure.

    Args:
        struc (Structure): Pymatgen Structure object.

    Returns:
        Structure: Structure object with charges removed, or None if input is None.
    """
    # Check if structure is provided
    if struc:
        # Make a copy of the input structure to avoid modifying the original
        input_struc = struc.copy()

        # Iterate over each site in the structure
        for index, site in enumerate(input_struc.sites):
            # Remove charges from the species of each site
            new_site = site.species.remove_charges()

            # Replace the site in the structure with the modified site
            input_struc.replace(index, new_site)

        # Return the modified structure
        return input_struc
    else:
        # Print message if structure is missing
        print('Structure missing')
        return None


def simplify_structure(struc):
    """
    Simplifies the structure by replacing certain species with predefined species.

    Args:
        struc (Structure): Pymatgen Structure object to be simplified.

    Returns:
        Structure: Simplified Structure object, or None if input is None.
    """
    if struc:
        # Make a copy of the input structure to avoid modifying the original
        input_struc = struc.copy()

        # Initialize a list to store the simplified sites
        simplified_sites = []

        # Iterate over each site in the structure
        for idx, site in enumerate(input_struc.sites):
            # Get the species and their occupancies for the current site as a dictionary
            site_dic = input_struc.species_and_occu[idx].as_dict()

            # Iterate over each species and its label in the site dictionary
            for specie, specie_label in zip(site.species, list(site_dic.keys())):
                # Get the oxidation state of the species
                charge = specie.oxi_state

                # Extract the element symbol from the species label using regular expressions
                element_pattern = re.compile('[a-zA-Z]+')
                element = element_pattern.findall(specie_label)[0]

                # Skip Li species
                if element == 'Li':
                    continue
                else:
                    # Determine the replacement species based on the oxidation state
                    if charge == 0:
                        replacement_specie = Species("Mg", oxidation_state=charge)
                    elif charge > 0:
                        replacement_specie = Species("Al", oxidation_state=charge)
                    else:
                        replacement_specie = Species("S", oxidation_state=charge)

                    # Check if the replacement species is the same as the current species
                    if replacement_specie == specie:
                        continue
                    else:
                        # Check if the replacement species already exists in the site dictionary
                        if replacement_specie.to_pretty_string() in site_dic.keys():
                            # Add the occupancy of the current species to the replacement species
                            site_dic[replacement_specie.to_pretty_string()] += site_dic[specie_label]
                            # Remove the current species from the site dictionary
                            del site_dic[specie_label]
                        else:
                            # Replace the current species with the replacement species in the site dictionary
                            site_dic[replacement_specie.to_pretty_string()] = site_dic.pop(specie_label)

            # Create a simplified PeriodicSite object using the modified site dictionary
            simplified_sites.append(PeriodicSite(Composition(site_dic), site.frac_coords, lattice=input_struc.lattice))

        # Create a simplified Structure object from the simplified sites
        simplified_struc = Structure.from_sites(simplified_sites)

        return simplified_struc

    else:
        # Return None if the input structure is None
        return None


def construct_megnet_graph(struc, graph_converter, embeddings):
    """
    Construct a graph representation compatible with the megnet model from Pymatgen structures. 
    Linear combinations of elemental embeddigns are used to represent disordered compounds.

    Args:
        struc (Structure): Pymatgen Structure object.
        graph_converter: Graph converter object for converting structures to graphs.
        embeddings (np.array): Embeddings for elements.

    Returns:
        dict: Graph representation suitable for megnet model, or None if input is None.
    """
    try:
        # Check if structure is provided
        if struc:
            # Remove charges from the structure
            struc = remove_charges(struc)
            
            # Convert structure to graph representation
            graph = graph_converter.convert(struc)
            
            # Check if graph is valid
            if len(graph['bond']) == 0 or len(graph['atom']) == 0:
                print('Invalid graph for structure: ', struc.formula)
                return None
            else:
                # Calculate site embeddings
                site_embeddings = []
                for site in graph['atom']:
                    site_emb = 0
                    for element in site.keys():
                        element_emb = embeddings[Element(element).Z]
                        element_multiplier = site[element]
                        site_emb += element_emb * element_multiplier
                    site_embeddings.append(site_emb)
                
                # Update graph with site embeddings
                graph['atom'] = site_embeddings
                
                return graph
        else:
            return None
    except Exception as e:
        # Handle exception if cutoff is too small
        if "The cutoff is too small" in str(e):
            max_attempts = 10
            current_attempt = 0
            orig_cutoff = graph_converter.cutoff
            new_cutoff = orig_cutoff + 1
            while current_attempt <= max_attempts:
                print(f"Cutoff value is too small, increasing cutoff to {new_cutoff}")
                # Increase cutoff value and retry
                graph_converter = CrystalGraphDisordered(bond_converter=GaussianDistance(
                    np.linspace(0, new_cutoff + 1, int(np.round(100 * (new_cutoff + 1) / (orig_cutoff + 1)))), 0.5),
                                                         cutoff=new_cutoff)
                try:
                    # Remove charges from the structure
                    struc = remove_charges(struc)
                    
                    # Convert structure to graph representation
                    graph = graph_converter.convert(struc)
                    
                    # Check if graph is valid
                    if len(graph['bond']) == 0 or len(graph['atom']) == 0:
                        print('Invalid graph for structure: ', struc.formula)
                        return None
                    else:
                        # Calculate site embeddings
                        site_embeddings = []
                        for site in graph['atom']:
                            site_emb = 0
                            for element in site.keys():
                                element_emb = embeddings[Element(element).Z]
                                element_multiplier = site[element]
                                site_emb += element_emb * element_multiplier
                            site_embeddings.append(site_emb)
                        
                        # Update graph with site embeddings
                        graph['atom'] = site_embeddings
                        
                        return graph
                except Exception as e:
                    current_attempt += 1
                    new_cutoff += 1
                    continue
        else:
            return None


def get_megnet_feature(graph, describer):
    """
    Get atom features (V1, V2, V3) suitable for AtomSets model.

    Args:
        graph (dict): Graph representation of a structure.
        describer: Describer object for transforming graphs to features.

    Returns:
        pandas Dataframe: Site features suitable for megnet model, or None if input is None.
    """
    # Check if graph is provided
    if graph:
        # Transform graph to site features using describer
        return describer.transform([graph])[0]
    else:
        return None


def get_megnet_composition_feature(graph):
    """
    Get composition features (V0) suitable for AtomSets model.

    Args:
        graph (dict): Graph representation of a structure.
    Returns:
        pandas Dataframe: Composition features suitable for megnet model, or None if input is None.
    """
    # Check if graph is provided
    if graph:
        # Transform graph to site features using describer
        return pd.DataFrame(graph['atom'])
    else:
        return None


def find_duplicate_structures(df):
    """
    Find duplicate structures in a DataFrame of structures.

    Args:
        df (DataFrame): DataFrame containing structures.

    Returns:
        dict: Dictionary where keys are merged sets of indices representing duplicate structures,
              and values are sets of merged indices.
    """
    sm = StructureMatcher()  
    duplicates = {}  

    for i in range(len(df)):  
        struct1 = df.iloc[i]['structure']  
        struc_dic = str(struct1.as_dict())  
        if struc_dic not in duplicates:  
            duplicates[struc_dic] = [i]  
        for j in range(i + 1, len(df)):
            struct2 = df.iloc[j]['structure']  
            if sm.fit(struct1, struct2):
                if struc_dic not in duplicates:
                    duplicates[struc_dic] = [j] 
                duplicates[struc_dic].append(j)

    merged_duplicates = {}

    for key, indices in duplicates.items():
        sorted_indices = sorted(indices)
        if any(set(sorted_indices) & set(merged_indices) for merged_indices in merged_duplicates.values()):
            merged_indices_set = set()
            for merged_key, merged_indices in merged_duplicates.items():
                if set(sorted_indices) & set(merged_indices):
                    merged_indices_set.update(merged_indices)
                    merged_indices_set.update(sorted_indices)
                    merged_duplicates[merged_key] = list(merged_indices_set)
                    break
        else:
            merged_duplicates[tuple(sorted_indices)] = sorted_indices
    
    return merged_duplicates


def get_space_group(icsd, cif_dir):
    """
    Extract the space group number from a CIF file.

    Args:
        icsd (str): ICSD entry.
        cif_dir (str): Directory containing CIF files.

    Returns:
        int: The space group number if found, otherwise None.
    """
    # Construct path to CIF file
    cif_file_path = os.path.join(cif_dir, f"{icsd}.cif")

    # Check if CIF file exists
    if os.path.exists(cif_file_path):
        with open(cif_file_path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Check if the line contains the desired information
                if line.strip().startswith('_space_group_IT_number'):
                    # Extract numerical value from the line
                    numerical_value = line.split()[-1]
                    numerical_value = ''.join(filter(str.isdigit, numerical_value))
                    return int(numerical_value)
        
        # Print message if space group number is not found
        print(f'Space group number for {icsd} not found!')
        return None
    else:
        # Print message if CIF file is not found
        print(f'No CIF file found for ICSD entry {icsd}')
        return None


def get_el2md_mapper(formula_list):
    """
    Get a mapper object for mapping element to molecule distances.

    Args:
        formula_list (list): List of chemical formulas.

    Returns:
        mapper: Mapper object for element to molecule distances.
    """
    # Initialize the mapper object
    mapper = ElM2D()
    
    # Set the list of chemical formulas in the mapper object
    mapper.formula_list = formula_list

    # Calculate distances between pairs of chemical formulas
    n = len(formula_list)
    distances = []
    
    # Iterate over pairs of chemical formulas and calculate distances
    for i in tqdm(range(n - 1)):
        # Create an ElementMD object for the first formula
        x = ElMD(formula_list[i], metric=mapper.metric)
        
        # Calculate distances between the first formula and subsequent formulas
        for j in range(i + 1, n):
            distances.append(x.elmd(formula_list[j]))

    # Convert distances to a square distance matrix
    dist_vec = np.array(distances)
    mapper.dm = squareform(dist_vec)

    return mapper


def get_normalized_formula(structure):
    """
    Get the normalized formula of a structure.

    Args:
        structure (Structure): Pymatgen Structure object.

    Returns:
        str: Normalized formula of the structure.
    """
    # Create an ElementComposition object from the formula of the structure
    x = ElMD(structure.formula)
    
    # Get the pretty formula (normalized formula) from the ElementComposition object
    normalized_formula = x.pretty_formula

    return normalized_formula


def get_formula(structure):
    """
    Get the formula of a structure.

    Args:
        structure (Structure): Pymatgen Structure object.

    Returns:
        str: Formula of the structure.
    """
    
    return structure.formula.replace(" ", "")


def predict_band_gap(graph, model):
    """
    Predict the band gap of a material using a MEGNet and its graph representation.

    Args:
        graph (dict): Graph representation of a structure.
        model: MEGNet model for predicting band gaps.

    Returns:
        float: Predicted band gap if graph is valid, otherwise None.
    """
    if graph:
        return model.predict_graph(graph)[0]
    else:
        return None


def extract_band_gap_mp(icsd_code, mp_icsd_band_gap_dict):
    """
    Extract the band gap from a dictionary using the ICSD code.

    Args:
        icsd_code (str): ICSD collection code.
        mp_icsd_band_gap_dict (dict): Dictionary mapping ICSD codes to band gaps from the Materials Project.

    Returns:
        float: Band gap value if ICSD code is found in the dictionary, otherwise None.
    """
    if icsd_code in mp_icsd_band_gap_dict.keys():
        return mp_icsd_band_gap_dict[icsd_code]
    else:
        return None


def parse_formula(formula_str):
    if isinstance(formula_str, float):
        return {'element': 'Unknown', 'amount': formula_str}
    
    if any(char.isdigit() for char in formula_str):
        elements = formula_str.split()
        parsed_formula = {}
        for element in elements:
            match = re.match(r'([A-Za-z]+)(\d+(\.\d+)?)?', element)
            if match:
                element_name, element_amount, _ = match.groups()
                parsed_formula[element_name] = float(element_amount) if element_amount else 1.0
        return parsed_formula
    else:
        return {'element': formula_str, 'amount': 1.0}


def check_formulas_similar(formula1, formula2, tolerance=0.05):
    if set(formula1.keys()) != set(formula2.keys()):
        return False
    for element, amount1 in formula1.items():
        amount2 = formula2[element]
        if abs(amount1 - amount2) > tolerance:
            return False
    return True


def get_anion(struc):
    elements = [sp.symbol for sp in struc.composition.elements]
    if 'D' in elements:
        elements.remove('D')
    unique_elements = list(set(elements))
    # Find element with the greatest Pauling electronegativity, define this as the anion
    electronegativities = [Element(i).X for i in unique_elements]
    anion = unique_elements[electronegativities.index(max(electronegativities))]
    return anion


def get_electronegativity(anion):
    return pt.Element(anion).X


def generate_vacancy_structure(struc):
    """
    Generate a vacancy structure by adding dummy species to disordered sites where
    the occupancy sum is less than 1.

    Args:
        struc (pymatgen.core.structure.Structure): The structure containing disordered sites.
    
    Returns:
        pymatgen.core.structure.Structure: A new structure where vacancies have been added to disordered sites.
    """
    extended_sites = []
    for i, site in enumerate(struc):
        if site.is_ordered:
            # Retain ordered sites
            extended_sites.append(site)
        else:
            # Handle disordered sites
            site_dic = struc.species_and_occu[i].as_dict()
            total_occu = sum(occu for occu in site_dic.values())
            if total_occu < 1:
                # Add a vacancy (DummySpecies) to account for missing occupancy
                site_dic[DummySpecies()] = 1 - total_occu
            extended_sites.append(
                PeriodicSite(Composition(site_dic), site.frac_coords, lattice=struc.lattice)
            )
    
    # Create a new structure from the modified sites
    return Structure.from_sites(extended_sites)


def generate_initial_structure(struc, initial_rc=2.0, rc_decrement=0.1):
    """
    Generate a random initial structure by selecting species from fractional occupancy 
    and validating based on a distance cutoff.

    Args:
        struc (pymatgen.core.structure.Structure): The input structure to modify.
        initial_rc (float): Initial distance cutoff for atom separation.
        rc_decrement (float): Decrement value for the cutoff after each unsuccessful attempt.

    Returns:
        pymatgen.core.structure.Structure: A valid randomized structure where atoms respect distance constraints.
    """
    rc = initial_rc
    vacancy_struc = generate_vacancy_structure(struc)
    species_and_occu = [i.species.as_dict() for i in vacancy_struc.sites]
    attempt = 0

    while True:
        # Generate a random structure based on site occupation probabilities
        init_struc = generate_random_structure(vacancy_struc, species_and_occu)

        # Validate structure based on distance constraints
        if check_structure_validity(init_struc, rc):
            return init_struc

        # Decrease the distance cutoff after every 200 attempts
        if attempt % 200 == 199:
            rc -= rc_decrement
            print(f"Decrementing rc to {rc}")

        attempt += 1


def generate_random_structure(vacancy_struc, species_and_occu):
    """
    Generate a random structure by assigning species based on their occupancy probabilities.

    Args:
        vacancy_struc (pymatgen.core.structure.Structure): The vacancy structure used to sample species.
        species_and_occu (list): List of species and their occupancy probabilities.

    Returns:
        pymatgen.core.structure.Structure: A randomized structure with species assigned based on occupancy.
    """
    random_structure_sites = []

    for site, occu_dict in zip(vacancy_struc.sites, species_and_occu):
        if len(occu_dict) == 1:
            # Retain sites with a single species
            random_structure_sites.append(site)
        else:
            # Select species based on occupancy probabilities
            chosen_species = choose_species_based_on_occupancy(occu_dict)
            random_structure_sites.append(
                PeriodicSite(Composition({chosen_species: 1}), site.frac_coords, lattice=vacancy_struc.lattice)
            )

    return Structure.from_sites(random_structure_sites)


def choose_species_based_on_occupancy(occupancy_dict):
    """
    Randomly select a species based on its occupancy probability.

    Args:
        occupancy_dict (dict): Dictionary of species and their corresponding occupancy.

    Returns:
        str: Selected species based on occupancy.
    """
    species = list(occupancy_dict.keys())
    probabilities = list(occupancy_dict.values())

    # Randomly select a species based on the probabilities
    return np.random.choice(species, p=probabilities)


def check_structure_validity(init_struc, rc):
    """
    Validate the generated structure by checking the minimum distance between atoms 
    and ensuring lithium (Li) is present.

    Args:
        init_struc (pymatgen.core.structure.Structure): The structure to validate.
        rc (float): The minimum acceptable distance between atoms.

    Returns:
        bool: True if the structure is valid, False otherwise.
    """
    val_struc = init_struc.copy(site_properties=None)
    val_struc.remove_sites([i for i, site in enumerate(val_struc.sites) if site.species.valid is False])

    diag = np.eye(len(val_struc.distance_matrix), dtype=bool)
    distance_check = not np.any(val_struc.distance_matrix[~diag] <= rc)
    li_present_check = 'Li' in val_struc.symbol_set

    # Return True if distances are valid and Li is present
    return distance_check and li_present_check


def generate_trial_structure(struc, init_struc, max_attempts=500):
    """
    Generate a trial structure by swapping species between disordered sites and checking 
    if the swap is valid.

    Args:
        struc (pymatgen.core.structure.Structure): The original structure.
        init_struc (pymatgen.core.structure.Structure): The initial randomized structure.
        max_attempts (int): Maximum number of swap attempts.

    Returns:
        pymatgen.core.structure.Structure or None: The new structure if a valid swap occurs, 
        or None if no valid swap is found.
    """
    disordered_sites = [i for i, site in enumerate(struc.sites) if not site.is_ordered]
    vacancy_struc = generate_vacancy_structure(struc)
    attempts = 0

    while attempts < max_attempts:
        attempts += 1

        # Randomly select two disordered sites
        site1, site2 = random.sample(disordered_sites, 2)

        site1_init_identity = init_struc.sites[site1].species_string
        site2_init_identity = init_struc.sites[site2].species_string

        # Ensure the selected sites have different species and can be swapped
        if site1_init_identity != site2_init_identity and check_can_swap(vacancy_struc, init_struc, site1, site2, site1_init_identity, site2_init_identity):
            # Calculate swap probability and decide to swap
            p_accept = calculate_swap_probability(vacancy_struc, site1, site2, site1_init_identity, site2_init_identity)
            if np.random.uniform() <= p_accept:
                return swap_sites(init_struc, site1, site2)

    return None


def check_can_swap(vacancy_struc, initial_struc, site1, site2, site1_init_identity, site2_init_identity):
    """
    Check if two sites can be swapped based on their valid species in the vacancy structure.

    Args:
        vacancy_struc (pymatgen.core.structure.Structure): The vacancy structure.
        initial_struc (pymatgen.core.structure.Structure): The initial randomized structure.
        site1 (int): Index of the first site.
        site2 (int): Index of the second site.
        site1_init_identity (str): Species identity of the first site.
        site2_init_identity (str): Species identity of the second site.

    Returns:
        bool: True if the sites can be swapped, False otherwise.
    """
    valid_elems_site1 = [re.match(r"([A-Z][a-z]*)", spec).group() for spec in vacancy_struc.species_and_occu[site1].as_dict().keys()]
    valid_elems_site2 = [re.match(r"([A-Z][a-z]*)", spec).group() for spec in vacancy_struc.species_and_occu[site2].as_dict().keys()]

    site1_element = re.match(r"([A-Z][a-z]*)", site1_init_identity).group()
    site2_element = re.match(r"([A-Z][a-z]*)", site2_init_identity).group()

    # Check if the species can be swapped between the two sites
    return site1_element in valid_elems_site2 and site2_element in valid_elems_site1


def calculate_swap_probability(vacancy_struc, site1, site2, site1_init_identity, site2_init_identity):
    """
    Calculate the probability of swapping two species based on their occupancy.

    Args:
        vacancy_struc (pymatgen.core.structure.Structure): The vacancy structure.
        site1 (int): Index of the first site.
        site2 (int): Index of the second site.
        site1_init_identity (str): Species identity of the first site.
        site2_init_identity (str): Species identity of the second site.

    Returns:
        float: Probability of accepting the swap.
    """
    species_and_occu = [i.species.as_dict() for i in vacancy_struc.sites]
    p_atom1_site1 = species_and_occu[site1][site1_init_identity]
    p_atom1_site2 = species_and_occu[site2][site1_init_identity]
    p_atom2_site1 = species_and_occu[site1][site2_init_identity]
    p_atom2_site2 = species_and_occu[site2][site2_init_identity]

    # Return swap probability based on occupancy ratios
    return (p_atom1_site2 * p_atom2_site1) / (p_atom1_site1 * p_atom2_site2)


def swap_sites(init_struc, site1, site2):
    """
    Perform the actual swapping of two species in the structure.

    Args:
        init_struc (pymatgen.core.structure.Structure): The structure before swapping.
        site1 (int): Index of the first site.
        site2 (int): Index of the second site.

    Returns:
        pymatgen.core.structure.Structure: The structure with the species swapped.
    """
    new_struc = init_struc.copy()

    # Swap the coordinates of the two sites
    new_struc[site1].coords = init_struc[site2].coords
    new_struc[site2].coords = init_struc[site1].coords

    return new_struc


def get_supercell_and_interior_cell_sites_ckdtree(struc, tolerance=1e-8):
    """
    Generate a 3x3x3 supercell and identify the interior sites that correspond to the fractional 
    coordinates of the original unit cell using a KD-tree for spatial matching.

    Args:
        struc (pymatgen.core.structure.Structure): The input structure (unit cell) to generate the supercell from.
        tolerance (float): The distance tolerance for identifying matching sites in the supercell.

    Returns:
        supercell_struc (pymatgen.core.structure.Structure): The generated 3x3x3 supercell.
        interior_sites_dic (dict): A dictionary mapping indices of interior sites in the supercell to their corresponding sites.
    """

    # Scale down the fractional coordinates of the original structure to fit within the supercell
    # These are the coordinates we expect to match with the interior of the supercell.
    interior_supercell_frac_coords = (struc.frac_coords + np.ones(3)) / 3

    # Create a 3x3x3 supercell by making a copy of the original structure and expanding it.
    supercell_struc = struc.copy()
    supercell_struc.make_supercell([3, 3, 3])

    # Build a KD-tree from the interior supercell fractional coordinates
    # This allows us to efficiently query the closest points in the supercell to match the original structure's sites.
    interior_kdtree = cKDTree(interior_supercell_frac_coords)

    # Initialize a dictionary to store the interior sites
    interior_sites_dic = {}

    # Loop over all sites in the 3x3x3 supercell and query the KD-tree to find matching sites
    for index, new_site in enumerate(supercell_struc.sites):
        # Find the closest match in the KD-tree for the new site's fractional coordinates
        _, i = interior_kdtree.query(new_site.frac_coords, distance_upper_bound=tolerance)

        # If the closest match is within tolerance and is an interior site, store it in the dictionary
        if i < len(interior_supercell_frac_coords):
            interior_sites_dic[index] = new_site

    # Return the 3x3x3 supercell and the dictionary of interior sites
    return supercell_struc, interior_sites_dic


def calculate_structure_features_optimized(struc):
    """
    Calculate various structural features from a given structure, including distances between Li-Li, 
    Li-anion, and anion-anion, as well as the average sublattice ionicity. The function also generates 
    a 3x3x3 supercell to account for periodic boundary conditions.

    Args:
        struc (pymatgen.core.structure.Structure): The input structure to calculate features from.

    Returns:
        list: A list of calculated structural features:
            - Average minimum Li-Li distance.
            - Average minimum Li-anion distance.
            - Average number of Li-Li bonds.
            - Average sublattice ionicity.
            - Average anion coordination.
    """
    
    # Create a copy of the structure and remove any invalid species
    val_struc = struc.copy(site_properties=None)
    val_struc.remove_sites([i for i, site in enumerate(val_struc.sites) if site.species.valid is False])

    # Get unique elements from the structure
    elements = [sp.symbol for sp in val_struc.species]
    unique_elements = list(set(elements))

    # Find the anion based on the highest Pauling electronegativity
    electronegativities = [pt.Element(i).X for i in unique_elements]
    anion = unique_elements[electronegativities.index(max(electronegativities))]

    # Generate a 3x3x3 supercell for periodic boundary conditions and get interior sites
    supercell_struc, interior_sites_dic = get_supercell_and_interior_cell_sites_ckdtree(val_struc)

    # Identify indices of Li, anion, and non-Li atoms in the supercell and interior cell
    interior_li_indices = [key for key in interior_sites_dic.keys() if interior_sites_dic[key].specie.element.value == 'Li']
    interior_anion_indices = [key for key in interior_sites_dic.keys() if interior_sites_dic[key].specie.element.value == anion]
    interior_non_li_indices = [key for key in interior_sites_dic.keys() if interior_sites_dic[key].specie.element.value != 'Li']
    
    # Find indices of Li and anion atoms in the entire supercell
    supercell_li_indices = np.where([species.element.value == 'Li' for species in supercell_struc.species])[0]
    supercell_anion_indices = np.where([species.element.value == anion for species in supercell_struc.species])[0]

    # Get the distance matrix of the supercell
    distances = supercell_struc.distance_matrix

    # Calculate average minimum Li-Li distance and average number of Li-Li bonds
    li_li_distances = distances[supercell_li_indices][:, interior_li_indices]
    avg_min_li_li_distance = np.mean(np.partition(li_li_distances, 1, axis=0)[1, :])  # Second smallest distance
    avg_num_li_li_bonds = np.mean(np.sum((li_li_distances > 0) & (li_li_distances <= 4), axis=0))  # Bonds within 4 Å

    # Calculate average minimum Li-anion distance
    li_anion_distances = distances[supercell_anion_indices][:, interior_li_indices]
    avg_min_li_anion_distance = np.mean(np.min(li_anion_distances, axis=0))

    # Calculate average anion-anion coordination (number of close anion neighbors)
    anion_anion_distances = distances[supercell_anion_indices][:, interior_anion_indices]
    closest_anion_distances = np.partition(anion_anion_distances, 1, axis=0)[1, :]  # Second smallest distance
    threshold_coordination_values = closest_anion_distances + 1  # Threshold for coordination (1 Å beyond minimum distance)
    avg_anion_coordination = np.mean(np.sum(anion_anion_distances < threshold_coordination_values, axis=0) - 1)

    # Calculate average sublattice ionicity (difference in electronegativity between atoms within 4 Å)
    all_electronegativities = np.array([pt.Element(sp.symbol).X for sp in supercell_struc.species])
    electronegativity_diff = np.abs(all_electronegativities[:, np.newaxis] - all_electronegativities)
    num_atoms = distances.shape[1]

    # Create a mask to only include non-Li atoms and exclude self-interactions
    mask = np.ones((num_atoms, num_atoms), dtype=bool)
    mask[interior_non_li_indices, :] = False
    mask = ~mask
    np.fill_diagonal(mask, False)
    prev_mask = mask.copy()
    mask = mask & ~mask.T
    mask = mask | np.triu(prev_mask)

    # Apply the mask to distances and electronegativity differences
    masked_distances = distances[mask]
    masked_electronegativity_diff = electronegativity_diff[mask]
    
    # Select pairs of atoms that are within 4 Å of each other
    indices = np.where(masked_distances < 4)
    avg_ionicity = np.mean(masked_electronegativity_diff[indices])

    # Return all calculated structural features
    return [avg_min_li_li_distance, avg_min_li_anion_distance, avg_num_li_li_bonds, avg_ionicity, avg_anion_coordination]


def extract_features_until_convergence(struc, convergence_threshold=0.01, consecutive_steps=100,
                                       max_init_attempts=500, max_outer_loop=1000, max_consecutive_no_change=100):
    """
    Extract features from a structure by generating trial structures and checking convergence 
    on the feature set. This process runs until the feature values converge or a maximum number 
    of iterations are reached.

    Args:
        struc (pymatgen.core.structure.Structure): The input structure from which to extract features.
        convergence_threshold (float): The threshold below which feature changes must stay for convergence.
        consecutive_steps (int): The number of consecutive steps where features must remain stable to reach convergence.
        max_init_attempts (int): Maximum number of initial trial structures generated before restarting the process.
        max_outer_loop (int): Maximum number of outer loops allowed to search for a valid structure.
        max_consecutive_no_change (int): Maximum number of consecutive steps where no significant change occurs, 
                                         used as a convergence criterion.

    Returns:
        numpy.ndarray: Array of averaged feature values after convergence, or None if convergence is not achieved.
    """
    
    # If the structure is ordered and does not contain Li, return None
    if struc.is_ordered:
        if 'Li' not in struc.symbol_set:
            print('No Li in structure!')
            return None

        # Calculate features directly for ordered structures
        features = calculate_structure_features_optimized(struc)
        return features

    else:
        # If the structure does not contain Li, return None
        if 'Li' not in struc.symbol_set:
            print('No Li in structure!')
            return None

        # List of disordered sites (sites with fractional occupancy)
        disordered_sites = [site for site in struc if not site.is_ordered]
        
        # If there is only one disordered site, expand the supercell along the shortest lattice vector
        if len(disordered_sites) == 1:
            supercell_multiplier = np.array([1, 1, 1])
            supercell_multiplier[np.argmin(struc.lattice.abc)] = 2
            struc.make_supercell(supercell_multiplier)

        # Generate a vacancy structure and initialize the loop variables
        vacancy_struc = generate_vacancy_structure(struc)
        outer_loop_count = 0
        consecutive_init_no_change = 0
        k = 1  # For tracking the number of outer loop iterations used in averaging

        # Outer loop: continue until convergence or max iterations are reached
        while outer_loop_count < max_outer_loop and consecutive_init_no_change < max_consecutive_no_change:
            print('outer loop count: ', outer_loop_count)
            print('consecutive no change: ', consecutive_init_no_change)

            trial_attempts = 0
            n = 1  # For tracking the number of inner loop attempts

            # Generate initial structure and calculate features
            init_struc = generate_initial_structure(vacancy_struc)
            running_average = np.array(calculate_structure_features_optimized(init_struc))

            # Inner loop: attempt swaps to optimize the trial structure
            while trial_attempts < max_init_attempts:
                trial_attempts += 1

                # Generate a trial structure by swapping disordered sites
                trial_struc = generate_trial_structure(struc, init_struc)
                if trial_struc:
                    break  # Break if a valid trial structure is found
                init_struc = generate_initial_structure(vacancy_struc)
                n += 1
                # Update running average of features
                running_average += (calculate_structure_features_optimized(init_struc) - running_average) / n

            prev_running_average = running_average.copy()

            # Initialize convergence tracking variables
            consecutive_steps_below_threshold = 0

            if trial_struc:
                # Calculate features for the new trial structure and update the running average
                n += 1
                running_average += (calculate_structure_features_optimized(trial_struc) - running_average) / n

                # Check for convergence by comparing current and previous averages
                if np.all(np.abs((running_average - prev_running_average) / (running_average + 1e-10)) <= convergence_threshold):
                    consecutive_steps_below_threshold += 1
                else:
                    consecutive_steps_below_threshold = 0

                # Continue adjusting trial structures until convergence
                while consecutive_steps_below_threshold < consecutive_steps:
                    # Generate more trial structures
                    trial_struc = generate_trial_structure(struc, trial_struc, max_attempts=10000)

                    # Calculate features for the trial structure and update running average
                    features = np.array(calculate_structure_features_optimized(trial_struc))
                    n += 1
                    prev_running_average = running_average.copy()
                    running_average += (features - running_average) / n

                    # Check for convergence
                    if np.all(np.abs((running_average - prev_running_average) / (running_average + 1e-10)) <= convergence_threshold):
                        consecutive_steps_below_threshold += 1
                    else:
                        consecutive_steps_below_threshold = 0

                print('Converged features: ', running_average)

                # Update initial running average if this is the first loop
                if outer_loop_count == 0:
                    init_running_average = running_average.copy()
                    print('init_running_average: ', init_running_average)
                else:
                    # Update the running average over multiple outer loops
                    k += 1
                    prev_init_running_average = init_running_average.copy()
                    print('previous_outer_running_average: ', init_running_average)
                    init_running_average += (running_average - init_running_average) / k
                    print('new_outer_running_average: ', init_running_average)

                    # Check for convergence over the outer loops
                    if np.all(np.abs((init_running_average - prev_init_running_average) / (init_running_average + 1e-10)) <= convergence_threshold):
                        consecutive_init_no_change += 1
                    else:
                        consecutive_init_no_change = 0

            else:
                # Handle case where no valid trial structures were generated
                print('No initial structures were generated with swappable sites')

                if outer_loop_count == 0:
                    init_running_average = running_average.copy()
                    print('init_running_average: ', init_running_average)
                else:
                    # Update the outer loop running average
                    k += 1
                    prev_init_running_average = init_running_average.copy()
                    init_running_average += (running_average - init_running_average) / k
                    print('init_running_average: ', init_running_average)

                    # Check for convergence between outer loops
                    if np.all(np.abs((init_running_average - prev_init_running_average) / (init_running_average + 1e-10)) <= convergence_threshold):
                        consecutive_init_no_change += 1
                    else:
                        consecutive_init_no_change = 0

            outer_loop_count += 1

        # Handle case where the maximum number of outer loops is reached without convergence
        if outer_loop_count == max_outer_loop:
            print('Features not converged after max outer loop iterations')
        else:
            print('Features converged successfully!')

        # Return the final averaged features
        print('Final averaged features: ', init_running_average)
        return init_running_average