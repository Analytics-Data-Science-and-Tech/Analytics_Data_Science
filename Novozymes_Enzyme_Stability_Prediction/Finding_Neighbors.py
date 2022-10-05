import pandas as pd
from biopandas.pdb import PandasPdb

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

def amino_acid_neighbors(pdb_file, k):
    
    """
    This function appends the k-neighbors aminoacids 
    for a given pbd file. 
    
    pdb_file: the file 
    k: the number of neighbors
    """
    
    ## Reading the file 
    pdb_df =  PandasPdb().read_pdb(pdb_file)

    ## Extracting atom 
    atom_df = pdb_df.df['ATOM']
    
    ## Aggregating based on residue
    residue_agg = pd.DataFrame(atom_df.groupby(['residue_name', 'residue_number'])[['x_coord', 'y_coord', 'z_coord']].mean())
    residue_agg['residue_name'] = residue_agg.index.get_level_values(0)
    residue_agg['residue_number'] = residue_agg.index.get_level_values(1)
    residue_agg = residue_agg.reset_index(drop = True)
    residue_agg = residue_agg[['residue_name', 'residue_number', 'x_coord', 'y_coord', 'z_coord']].sort_values(by = 'residue_number').reset_index(drop = True)
    
    ## Scaling x-y-z coordinates
    scaler = MinMaxScaler()
    residue_agg[['x_coord', 'y_coord', 'z_coord']] = pd.DataFrame(scaler.fit_transform(residue_agg[['x_coord', 'y_coord', 'z_coord']]), columns = ['x_coord', 'y_coord', 'z_coord'])
    
    ## Finding the k-neighbors
    nbrs = NearestNeighbors(n_neighbors = (k + 1)).fit(residue_agg[['x_coord', 'y_coord', 'z_coord']])
    distances, indices = nbrs.kneighbors(residue_agg[['x_coord', 'y_coord', 'z_coord']])
    
    neighbors_names = ['neighbor_' + str(i) for i in range(1, (k + 1))]
    neighbors = pd.DataFrame(indices[:,1:], columns = neighbors_names)
    
    ## Appending neighbors
    residue = pd.concat([residue_agg, neighbors], axis = 1) 
    
    return residue

                               