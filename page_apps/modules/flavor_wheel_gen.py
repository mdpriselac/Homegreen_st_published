import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np


#### Basic info abou the coffee flavors ####
coffee_flavors = {
    "Fruity": {
        "Berry": ["Blackberry", "Raspberry", "Blueberry", "Strawberry","Generic Berry"],
        "Dried Fruit": ["Raisin", "Prune", "Fig"],
        "Other Fruit": ["Coconut", "Cherry", "Pomegranate", "Pineapple", "Grape", "Apple", "Pear", "Peach","Banana","Tropical"],
        "Citrus Fruit": ["Grapefruit", "Orange", "Lemon", "Lime","Generic Citrus"]
    },
    "Sour/Fermented": {
        "Sour": ["Sour Aromatics", "Acetic Acid", "Butyric Acid", "Isovaleric Acid", "Citric Acid", "Malic Acid"],
        "Alcohol/Fermented": ["Winey", "Whiskey", "Fermented", "Overripe"]
    },
    "Green/Vegetative": {
        "Fresh": ["Olive Oil", "Raw", "Green/Vegetative", "Dark Green", "Vegetative", "Hay-like", "Herb-like"],
        "Beany": ["Beany"]
    },
    "Other": {
        "Papery/Musty": ["Stale", "Cardboard", "Papery", "Woody", "Moldy/Damp", "Musty/Dusty", "Musty/Earthy", "Animalic", "Meaty Brothy", "Phenolic"],
        "Chemical": ["Bitter", "Salty", "Medicinal", "Petroleum", "Skunky", "Rubber"]
    },
    "Roasted": {
        'Pipe Tobacco': ['Pipe Tobacco'],
        'Tobacco': ['Tobacco'],
        'Burnt': ['Acrid','Ashy','Smoky','Burnt'],
        'Cereal': ['Grain', 'Malt']
    },
    "Spices": {
        "Pungent": ["Pungent"],
        "Pepper": ["Pepper"],
        "Brown Spice": ["Anise", "Nutmeg", "Cinnamon", "Clove","Brown Spice"]
    },
    "Nutty/Cocoa": {
        "Nutty": ["Peanuts", "Hazelnut", "Almond","Generic Nutty"],
        "Cocoa": ["Cocoa", "Dark Chocolate", "Milk Chocolate"]
    },
    "Sweet": {
        "Brown Sugar": ["Molasses", "Maple Syrup", "Caramelized", "Honey"],
        "Vanilla": ["Vanilla", "Vanillin"],
        "Overall Sweet": ["Overall Sweet", "Sweet Aromatics"]
    },
    "Floral": {
        "Black Tea": ["Black Tea"],
        "Floral": ["Floral", "Chamomile", "Rose", "Jasmine", "Perfumed"]
    }
}

# Some background setting
## Color Palette
### How many colors do we need? To make the colors work we need to see which categories have most flavors and make sure they can all have at least one color per flavor.
family_flavor_counts = {}
for fam in coffee_flavors.keys():
    fam_counter = 1
    for gen in coffee_flavors[fam].keys():
        fam_counter += 1
        for spec in coffee_flavors[fam][gen]:
            fam_counter += 1
    family_flavor_counts[fam] = fam_counter
# The largest family has 28 flavors. There are 9 families Let's make a color palette with 9x28 flavors (to create more distinctness, we could expand the gap at the end to 40 or so)
total_colors_needed = 252
#Now we can create the palette
base_colors = sns.color_palette('husl',n_colors=total_colors_needed)
#Now we need to loop through the flavors and assign each flavor family a color, divide the colors evenly among the genus and species within the family
def assign_color_to_flav_families(flav_fam,color_pallette):
    flav_fam_dict = coffee_flavors[flav_fam]
    flav_genus_list = list(flav_fam_dict.keys())
    flav_species_list = [flav_fam_dict[gen] for gen in flav_genus_list]
    flat_species_list = []
    for spec_list in flav_species_list:
        for spec in spec_list:
            flat_species_list.append(spec)
    flat_flav_list = [flav_fam] + flav_genus_list + flat_species_list
    #print(len(flat_flav_list))
    color_idx_start = list(coffee_flavors.keys()).index(flav_fam) * 28
    
    color_idx_increment = int(round(28/len(flat_flav_list),0))
    #print(color_idx_increment)
    color_code_assignments = {}
    color_idx = color_idx_start
    for flav in flat_flav_list:
        #print(color_idx)
        color_code_assignments[flav] = color_pallette[color_idx]
        color_idx += color_idx_increment
    return color_code_assignments

master_color_assignments = {}
for fam in coffee_flavors.keys():
    master_color_assignments.update(assign_color_to_flav_families(fam,base_colors))
    
#Now that we have a dictinoary of colors we need to start calculating the geometry of the flavor wheel.
# We're goign to evenly space species at the outside of the wheel and then base the genus and flavor 
# sizes on the number of species they contain.

#for each family and genus count how many species each contains.
species_count_dict_fams = {}
species_count_dict_gens = {}
for cfam in coffee_flavors.keys():
    fam_spec_count = 0
    for cgen in coffee_flavors[cfam].keys():
        cgen_spec_count = len(coffee_flavors[cfam][cgen])
        species_count_dict_gens[cgen] = cgen_spec_count
        fam_spec_count += len(coffee_flavors[cfam][cgen])
    species_count_dict_fams[cfam] = fam_spec_count

## Now we need to know how many total species there are so we can caculate the size for each speices
total_species_counter = 0
for fam in coffee_flavors.keys():
    for gen in coffee_flavors[fam].keys():
        total_species_counter += len(coffee_flavors[fam][gen])
total_species_counter

##To help create the model we'll precaculate the start and end point for each family based on how
## many species they contain
species_width = 2 * np.pi / total_species_counter
##set the inital family width
first_family_start = 0
first_family = list(coffee_flavors.keys())[0]
first_family_species_count = species_count_dict_fams[first_family]
first_family_width = species_width * first_family_species_count
first_family_end = first_family_start + first_family_width
family_widths = {list(coffee_flavors.keys())[0]: {'start':first_family_start, 'width':first_family_width, 'end':first_family_end}}
for family in coffee_flavors.keys():
    #get the index of the family in the list of families
    family_idx = list(coffee_flavors.keys()).index(family)
    if family_idx==0:
        print(f"{family} is the first family")
    else:
        fam_start = family_widths[list(coffee_flavors.keys())[family_idx-1]]['end']
        fam_species_count = species_count_dict_fams[family]
        fam_width = species_width * fam_species_count
        fam_end = fam_start + fam_width
        family_widths[family] = {'start':fam_start, 'width':fam_width, 'end':fam_end}

##We can then dynamically calculate the start for each genus
def genus_start_finder(fam,gen):
    
    #get the index of the genus in the list of genuses
    genus_idx = list(coffee_flavors[fam].keys()).index(gen)
    if genus_idx == 0:
        gen_start = family_widths[fam]['start']
    else:
        gen_start = family_widths[fam]['start'] + (species_width * sum([species_count_dict_gens[genus] for genus in coffee_flavors[fam].keys()][:genus_idx]))
    return gen_start 

# To dynamically calculate the alpha for each flavor present we're going to need to prep the data
# This function will take in the raw string input from the dataframe with the taxonomized flavors and return a dictionary with alpha levels
# for each flavor space on the wheel

##First, here's the funcitno for processing a list of dictionaries
def flavor_alphas(taxonomized_flavor_list_in):
    fam_counts = {}
    gen_counts = {}
    spec_counts = {}
    total_flavs = len(taxonomized_flavor_list_in)
    #First, let's count the mentions of each flavor family, genus and species
    for flav_dict in taxonomized_flavor_list_in:
        fam = flav_dict['family']
        gen = flav_dict['genus']
        spec = flav_dict['species']
        
        if fam in fam_counts:
            fam_counts[fam] += 1
        else:
            fam_counts[fam] = 1
        
        if gen in gen_counts:
            gen_counts[gen] += 1
        else:
            gen_counts[gen] = 1
        
        if spec in spec_counts:
            spec_counts[spec] += 1
        else:
            spec_counts[spec] = 1
    
    #Second, let's set an alpha level for each family, genus, and species based on the count for each and dividing by the total
    #number of flavor mentions
    fam_alpha = {fam: max(.5,fam_counts[fam]/total_flavs) for fam in fam_counts}
    gen_alpha = {gen: max(.5,gen_counts[gen]/total_flavs) for gen in gen_counts}
    spec_alpha = {spec: max(.5,spec_counts[spec]/total_flavs) for spec in spec_counts}
    
    return {'family_alphas':fam_alpha, 'genus_alphas':gen_alpha, 'species_alphas':spec_alpha}

## Now we can wrap this in something that takesa string input
def final_alpha_setter(tax_flav_str_in):
    cleaned_tax_str = tax_flav_str_in.replace("'",'"').replace('None','"None"')
    tax_flav_list = json.loads(cleaned_tax_str)
    
    return flavor_alphas(tax_flav_list)

# Now we have the geomtry information and the alpha information we can start to build a dynamic flavor wheel.

def flavor_wheel_gen(tax_flav_str_in):
    alpha_dict = final_alpha_setter(tax_flav_str_in)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    coffee_fams = list(coffee_flavors.keys())
    coffee_gens = {fam: list(coffee_flavors[fam].keys()) for fam in coffee_fams}
    coffee_specs = {gen: [spec for sublist in coffee_flavors[gen].values() for spec in sublist] for gen in coffee_gens}
    spec_width = 2 * np.pi / total_species_counter

    for fam in coffee_fams:
        family_width = family_widths[fam]['width']
        fam_start = family_widths[fam]['start']
        fam_end = family_widths[fam]['end']
        family_color = master_color_assignments[fam]
        family_alpha = alpha_dict['family_alphas'].get(fam,0)
        fam_mid = fam_start + (family_width / 2)
        # Plot family segment (inner circle)
        ax.bar([fam_start], [0.3], width=family_width, bottom=0,
                color=family_color, alpha=family_alpha, edgecolor='white', linewidth=1, align='edge')
        if 90 < np.degrees(fam_mid) < 270:
                fam_rot_degrees = np.degrees(fam_mid) +180
        else:
                fam_rot_degrees = np.degrees(fam_mid)
        ax.text(fam_start + (family_width / 2), 0.2, fam, ha='center', va='center', ma='center',rotation=fam_rot_degrees,color='white',fontsize=5)
        for gen in coffee_flavors[fam]:
                gen_spec_count = species_count_dict_gens[gen]
                gen_width = gen_spec_count * spec_width
                gen_start = genus_start_finder(fam,gen)
                genus_alpha = alpha_dict['genus_alphas'].get(gen,0)
                gen_color = master_color_assignments[gen]
                gen_mid = gen_start + gen_width/2
                # Plot genus segment (middle ring)
                ax.bar([gen_start], [0.4], width=gen_width, bottom=0.3,
                        color=gen_color, alpha=genus_alpha, edgecolor='white', linewidth=.5, align='edge')
                if 90 < np.degrees(gen_mid) < 270:
                        gen_rot_degrees = np.degrees(gen_mid) +180
                else:
                        gen_rot_degrees = np.degrees(gen_mid)       
                ax.text(gen_mid, 0.5, gen, ha='center', va='center', ma='center',rotation=gen_rot_degrees,color='white',fontsize=7)
                
                for spec in coffee_flavors[fam][gen]:
                        spec_start = gen_start + coffee_flavors[fam][gen].index(spec) * spec_width
                        spec_alpha = alpha_dict['species_alphas'].get(spec,0)
                        if spec in alpha_dict['species_alphas'].keys():
                            spec_color = master_color_assignments[spec]
                        else:
                            spec_color = 'None'
                        mid_spec = spec_start + spec_width/2
                        # Plot species segment (outer ring)
                        ax.bar([spec_start], [0.3], width=spec_width, bottom=0.7,
                                color=spec_color, alpha=spec_alpha, edgecolor='white', linewidth=0.5, align='edge')
                        if 90 < np.degrees(mid_spec) < 270:
                                spec_rot_degrees = np.degrees(mid_spec) +180
                        else:
                                spec_rot_degrees = np.degrees(mid_spec)
                        ax.text(mid_spec, .85, spec, ha='center', va='center', ma='center',rotation=spec_rot_degrees,color='white',fontsize=6)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.spines['polar'].set_visible(False)
    fig.patch.set_alpha(0)
    
    return fig