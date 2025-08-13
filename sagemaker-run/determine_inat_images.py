import pandas as pd

def main():

	# Baltimore lat lon
	balto_min_lat = 39.19
	balto_max_lat = 39.37
	balto_min_lon = -1.*76.71
	balto_max_lon = -1.*76.52

	# Maryland lat lon
	maryland_min_lat = 37.911717
	maryland_max_lat = 39.723043
	maryland_min_lon = -1.*79.487651
	maryland_max_lon = -1.*75.048939

	target_species = ['Lycorma delicatula', 'Zenaida macroura', 'Acer rubrum', 'Erigeron canadensis'] #Update this if you need other species

	acer_like_species = ["Acer saccharum", "Acer saccharinum", "Acer platanoides"] #This can be empty if you don't have the problem like Acer does

	species_to_get_from_md = target_species + acer_like_species

	# Determine which observations are from Maryland
	chunk_size = 10000
	maryland_observations = []
	for chunk_number, chunk in enumerate(pd.read_csv('./observations.csv', chunksize=chunk_size, sep = '\t')):

	    filtered_chunk = chunk.query(f"(latitude <= {maryland_max_lat}) & (latitude >= {maryland_min_lat}) & (longitude <= {maryland_max_lon}) & (longitude >= {maryland_min_lon})")

	    if not filtered_chunk.empty:
	        print(f"Appending {len(filtered_chunk)} records")
	        maryland_observations.append(filtered_chunk)
	maryland_obs_df = pd.concat(maryland_observations, ignore_index=True)

	taxa_df = pd.read_csv('./taxa.csv', index_col=None, sep='\t')
	maryland_obs_df = pd.merge(maryland_obs_df, taxa_df[["taxon_id", "name"]], on = "taxon_id")


	# Determine which observations are from Baltimore
	balto_obs_df = maryland_obs_df.query(f"(latitude <= {balto_max_lat}) & (latitude >= {balto_min_lat}) & (longitude <= {balto_max_lon}) & (longitude >= {balto_min_lon})")

	# Relevant observations from Maryland
	maryland_obs_uuids = maryland_obs_df[maryland_obs_df.name.isin(species_to_get_from_md)]["observation_uuid"].unique()

	# Determine which images we need from all of Maryland
	chunk_size = 10000
	maryland_uuid_photo_id_ext = []
	for chunk_number, chunk in enumerate(pd.read_csv('./photos.csv', chunksize=chunk_size, sep = '\t')):

	    filtered_chunk = chunk[chunk["observation_uuid"].isin(maryland_obs_uuids)]

	    if not filtered_chunk.empty:
	        unique_uuid_photo_id_ext = filtered_chunk[["observation_uuid", 'photo_id', 'extension']].drop_duplicates()
	        uuid_photo_ids_ext = list(zip(unique_uuid_photo_id_ext['observation_uuid'], unique_uuid_photo_id_ext['photo_id'], unique_uuid_photo_id_ext['extension']))

	        print(f"Appending {len(uuid_photo_ids_ext)} records")
	        
	        maryland_uuid_photo_id_ext += uuid_photo_ids_ext

	maryland_obs_with_photos =\
	pd.DataFrame(maryland_uuid_photo_id_ext,\
	             columns = ["observation_uuid", "photo_id", "extension"]).merge(maryland_obs_df[['observation_uuid', 'name']], on = 'observation_uuid')

	maryland_obs_with_photos["image_name"] = maryland_obs_with_photos.apply(lambda x: str(x["photo_id"]) + "_medium." + x["extension"], axis = 1)

	# Relevant observations from Baltimore
	balto_obs_uuids = balto_obs_df[~(balto_obs_df.name.isin(species_to_get_from_md))]["observation_uuid"].unique()

	# Determine which images we need from Baltimore
	chunk_size = 10000
	balto_uuid_photo_id_ext = []
	for chunk_number, chunk in enumerate(pd.read_csv('./photos.csv', chunksize=chunk_size, sep = '\t')):

	    filtered_chunk = chunk[chunk["observation_uuid"].isin(balto_obs_uuids)]

	    if not filtered_chunk.empty:
	        unique_uuid_photo_id_ext = filtered_chunk[["observation_uuid", 'photo_id', 'extension']].drop_duplicates()
	        uuid_photo_ids_ext = list(zip(unique_uuid_photo_id_ext['observation_uuid'], unique_uuid_photo_id_ext['photo_id'], unique_uuid_photo_id_ext['extension']))

	        print(f"Appending {len(uuid_photo_ids_ext)} records")
	        
	        balto_uuid_photo_id_ext += uuid_photo_ids_ext

	balto_obs_with_photos =\
	pd.DataFrame(balto_uuid_photo_id_ext,\
	             columns = ["observation_uuid", "photo_id", "extension"]).merge(balto_obs_df[['observation_uuid', 'name']], on = 'observation_uuid')

	balto_obs_with_photos["image_name"] = balto_obs_with_photos.apply(lambda x: str(x["photo_id"]) + "_medium." + x["extension"], axis = 1)


	# Combine Maryland observations with Baltimore observations, downsampling the latter to 10%
	combo_subsample = pd.concat([balto_obs_with_photos.sample(frac=0.1).rename(columns =\
	                                                                           {"name": "species_name"})[['species_name', 'image_name', 'photo_id', 'extension']].drop_duplicates(),\
	                             maryland_obs_with_photos.rename(columns =\
	                                                             {"name": "species_name"})[['species_name', 'image_name', 'photo_id', 'extension']].drop_duplicates()], ignore_index=True)

	combo_subsample[combo_subsample.duplicated(keep=False)].drop_duplicates()

	combo_subsample = combo_subsample[~combo_subsample.duplicated(subset=['image_name', 'photo_id', 'extension'], keep=False)]

	combo_subsample.to_csv("./training_images.csv", index=None)

if __name__ == "__main__":
	main()
