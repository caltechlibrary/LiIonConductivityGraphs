import os
import re
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure, Species, Element, PeriodicSite, Composition
from pymatgen.core import periodic_table as pt
from pymatgen.analysis.structure_matcher import StructureMatcher

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