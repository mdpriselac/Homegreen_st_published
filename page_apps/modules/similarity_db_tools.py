import pandas as pd
import streamlit as st


key_cols = ['uid', 'Name', 'country_final', 'subregion_final','categorized_flavors', 'process_type', 'fermentation']

## Creating the basic Flavor relevant data
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
        "Floral": ["Floral", "Rose", "Chamomile", "Rose", "Jasmine", "Perfumed"]
    }
}

family_genus_flavor_dict = {key: list(value.keys()) for key, value in coffee_flavors.items() if isinstance(value, dict)}
flavor_families = list(family_genus_flavor_dict.keys())
family_genus_flavors = {key: list(value.keys()) for key, value in coffee_flavors.items() if isinstance(value, dict)}

genus_lists = [genus_list for genus_list in family_genus_flavors.values()]
#unpack each list in genus_lists
genus_list = [genus for sublist in genus_lists for genus in sublist]

species_list = []
for family in flavor_families:
    family_genus_list = coffee_flavors[family].keys()
    for genus in family_genus_list:
        species_list.extend(coffee_flavors[family][genus])

species_list_no_dup = [spec for spec in species_list if spec not in genus_list]
corrected_spec = [spec + ' (Species)' for spec in species_list if spec in genus_list]
species_list_no_dup.extend(corrected_spec)


# Functions for manipulating particular rows/inputs
def create_flavor_dimension(flavor_tax_list_in):
    family_counts = {}
    genus_counts = {}
    species_counts = {}
    num_flavs = len(flavor_tax_list_in)
    for flavor_tax in flavor_tax_list_in:
        fam = flavor_tax['family']
        gen = flavor_tax['genus']
        spec = flavor_tax['species']
        if fam not in family_counts.keys():
            family_counts[fam] = 1/num_flavs
        else:
            family_counts[fam] += 1/num_flavs
        if gen not in genus_counts.keys():
            genus_counts[gen] = 1/num_flavs
        else:
            genus_counts[gen] += 1/num_flavs
        
        if spec not in species_counts.keys():
            species_counts[spec] = 1/num_flavs
        else:
            species_counts[spec] += 1/num_flavs
    
            
    return {'family_counts': family_counts, 'genus_counts': genus_counts, 'species_counts': species_counts}

def encode_flavor_info(pre_ohe_db_row_in,value_lists):
    flavor_family_list = value_lists['flavor_family_list']
    flavor_genus_list = value_lists['flavor_genus_list']
    flavor_species_list = value_lists['flavor_species_list']
    #encode the flavor family
    coffee_family_dict = {family: 0 for family in flavor_family_list}
    for flavor in pre_ohe_db_row_in['summed_flavorcounts']['family_counts'].keys():
        coffee_family_dict[flavor] = pre_ohe_db_row_in['summed_flavorcounts']['family_counts'][flavor]
    #ecode the flavor genus
    coffee_genus_dict = {genus: 0 for genus in flavor_genus_list}
    for flavor in pre_ohe_db_row_in['summed_flavorcounts']['genus_counts'].keys():
        coffee_genus_dict[flavor] = pre_ohe_db_row_in['summed_flavorcounts']['genus_counts'][flavor]
    #encode the flavor species
    coffee_species_dict = {species: 0 for species in flavor_species_list}
    for flavor in pre_ohe_db_row_in['summed_flavorcounts']['species_counts'].keys():
        if flavor in genus_list:
            new_flavor = flavor + ' (Species)'
        else:
            new_flavor = flavor
        coffee_species_dict[new_flavor] = pre_ohe_db_row_in['summed_flavorcounts']['species_counts'][flavor]
    coffee_name_dict = {'Name':pre_ohe_db_row_in['Name']}
    coffee_uid_dict = {'uid':pre_ohe_db_row_in['uid']}
    #concatenate all the dictionaries
    coffee_info_dict = {**coffee_uid_dict,**coffee_name_dict,**coffee_family_dict, **coffee_genus_dict, **coffee_species_dict}
    return coffee_info_dict


#Encoding functions to OHE coffee info
def encode_all_coffee_info(pre_ohe_db_row_in,value_lists):
    country_list = value_lists['country_list']
    subregion_list = value_lists['subregion_list']
    process_type_list = value_lists['process_type_list']
    flavor_family_list = value_lists['flavor_family_list']
    flavor_genus_list = value_lists['flavor_genus_list']
    flavor_species_list = value_lists['flavor_species_list']
    
    #encode the country
    coffee_country_dict = {country: 0 for country in country_list}
    coffee_country_dict[pre_ohe_db_row_in['country_final']] = 1
    #encode the subregion
    coffee_subregion_list = pre_ohe_db_row_in['subregion_list']
    coffee_subregion_dict = {subregion: 0 for subregion in subregion_list}
    for subr in coffee_subregion_list:
        if subr != 'UNKNOWN':
            coffee_subregion_dict[subr] = 1
    #encode the process type
    coffee_process_dict = {process: 0 for process in process_type_list}
    if pre_ohe_db_row_in['process_type'] != 'UNKNOWN':
        coffee_process_dict[pre_ohe_db_row_in['process_type']] = 1
    #encode the fermentation
    coffee_fermentation_dict = {'fermented': 0}
    if pre_ohe_db_row_in['fermentation']:
        coffee_fermentation_dict['fermented'] = 1
    #encode the flavor family
    coffee_family_dict = {family: 0 for family in flavor_family_list}
    for flavor in pre_ohe_db_row_in['summed_flavorcounts']['family_counts'].keys():
        coffee_family_dict[flavor] = pre_ohe_db_row_in['summed_flavorcounts']['family_counts'][flavor]
    #ecode the flavor genus
    coffee_genus_dict = {genus: 0 for genus in flavor_genus_list}
    for flavor in pre_ohe_db_row_in['summed_flavorcounts']['genus_counts'].keys():
        coffee_genus_dict[flavor] = pre_ohe_db_row_in['summed_flavorcounts']['genus_counts'][flavor]
    #encode the flavor species
    coffee_species_dict = {species: 0 for species in flavor_species_list}
    for flavor in pre_ohe_db_row_in['summed_flavorcounts']['species_counts'].keys():
        if flavor in genus_list:
            new_flavor = flavor + ' (Species)'
        else:
            new_flavor = flavor
        coffee_species_dict[new_flavor] = pre_ohe_db_row_in['summed_flavorcounts']['species_counts'][flavor]
    coffee_name_dict = {'Name':pre_ohe_db_row_in['Name']}
    coffee_uid_dict = {'uid':pre_ohe_db_row_in['uid']}
    #concatenate all the dictionaries
    coffee_info_dict = {**coffee_uid_dict,**coffee_name_dict,**coffee_country_dict, **coffee_subregion_dict, **coffee_process_dict, **coffee_fermentation_dict, **coffee_family_dict, **coffee_genus_dict, **coffee_species_dict}
    return coffee_info_dict

def trim_df(df):
    trimmed_db = df[key_cols].copy()
    #create subregion list
    trimmed_db['subregion_list'] = trimmed_db['subregion_final'].apply(lambda x: eval(x) if not pd.isnull(x) else x)
    #create flavor list
    trimmed_db['categorized_flavors_list'] = trimmed_db['categorized_flavors'].apply(lambda x: eval(x) if not pd.isnull(x) else x)
    #list of countries
    country_list = trimmed_db['country_final'].unique()
    country_list = [country for country in country_list if not pd.isnull(country)]
    #list of sub regions
    subregion_list = [subr for subr in trimmed_db['subregion_list'].explode().unique() if not pd.isnull(subr) and subr != "UNKNOWN"]
    #list of processes (ignoring unknown)
    process_type_list = trimmed_db['process_type'].unique()
    process_type_list = [proc for proc in process_type_list if not pd.isnull(proc) and 'unkn' not in proc.lower() and 'llm' not in proc.lower()]
    trimmed_db['summed_flavorcounts'] = trimmed_db['categorized_flavors_list'].apply(create_flavor_dimension)
    
    #drop subregion_final, categorized_flavors columns
    trimmed_db.drop(columns=['subregion_final','categorized_flavors'],inplace=True)
    #drop rows that are null in name, country_final, subregion_list, process_type, or categorized_flavors_list
    trimmed_db.dropna(subset=['uid','Name','country_final','subregion_list','process_type','categorized_flavors_list'],inplace=True)
    
    return trimmed_db, {'country_list': country_list, 'subregion_list': subregion_list, 'process_type_list': process_type_list, 'flavor_family_list': flavor_families, 'flavor_genus_list': genus_list, 'flavor_species_list': species_list_no_dup}

def ohe_df_all(df, value_lists):
    ohe_db = pd.DataFrame(df.apply(lambda x: encode_all_coffee_info(x,value_lists),axis=1).to_list())
    ohe_cols = ["Name", "uid"]
    for val_list in value_lists.values():
        ohe_cols.extend(val_list)
    #keep only the columns we want
    ohe_db_clean_cols = ohe_db[ohe_cols].copy()
    #drop any nulls
    ohe_db_clean_cols.dropna(inplace=True)
    
    return ohe_db_clean_cols

def ohe_df_flavors(df, value_lists):
    ohe_db = pd.DataFrame(df.apply(lambda x: encode_flavor_info(x,value_lists),axis=1).to_list())
    ohe_cols = ["Name", "uid"]
    #narrow down only the flavor value lists
    flav_val_lists = ['flavor_family_list','flavor_genus_list','flavor_species_list']
    value_lists_flavs_only = {key: value_lists[key] for key in value_lists.keys() if key in flav_val_lists}
    for val_list in value_lists_flavs_only.values():
        ohe_cols.extend(val_list)
    #keep only the columns we want
    ohe_db_clean_cols = ohe_db[ohe_cols].copy()
    #drop any nulls
    ohe_db_clean_cols.dropna(inplace=True)
    
    return ohe_db_clean_cols


def create_full_similarity_db(full_df_in,full_coll_conn):
    
    trimmed_db, value_lists = trim_df(full_df_in)
    ohe_db = ohe_df_all(trimmed_db, value_lists)
    
    uids = ohe_db['uid'].values.tolist()
    ignore_cols = ['Name','uid']
    relevant_cols = [col for col in ohe_db.columns if col not in ignore_cols]
    embeds = ohe_db[relevant_cols].values.tolist()
    mdata = ohe_db[['Name','uid']].to_dict('records')
    
    full_coll_conn.add(
        ids=uids,
        embeddings=embeds,
        metadatas=mdata
    )


def find_full_similarity_matches(coffee_id,original_df,full_coll_conn,num_matches=11):
    results = full_coll_conn.query(
        query_embeddings=full_coll_conn.get(ids=[coffee_id],include=["embeddings"])["embeddings"],
        n_results=num_matches
    )
    final_ids = results['ids'][0]
    final_ids = [match for match in final_ids if match != coffee_id]
    match_df = original_df[original_df['uid'].isin(final_ids)]
    return match_df


def create_flavor_similarity_db(full_df_in,flav_coll_conn):
    
    trimmed_db, value_lists = trim_df(full_df_in)
    ohe_db = ohe_df_flavors(trimmed_db, value_lists)
    
    uids = ohe_db['uid'].values.tolist()
    ignore_cols = ['Name','uid']
    relevant_cols = [col for col in ohe_db.columns if col not in ignore_cols]
    embeds = ohe_db[relevant_cols].values.tolist()
    mdata = ohe_db[['Name','uid']].to_dict('records')
    
    flav_coll_conn.add(
        ids=uids,
        embeddings=embeds,
        metadatas=mdata
    )

def find_flavor_similarity_matches(coffee_id,original_df,flav_coll_conn,num_matches=11):
    results = flav_coll_conn.query(
        query_embeddings=flav_coll_conn.get(ids=[coffee_id],include=["embeddings"])["embeddings"],
        n_results=num_matches
    )
    final_ids = results['ids'][0]
    final_ids = [match for match in final_ids if match != coffee_id]
    match_df = original_df[original_df['uid'].isin(final_ids)]
    return match_df

