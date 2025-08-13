# Training data collection

## Overview

Training data for this model comes from [iNaturalist](https://www.inaturalist.org/), which has a crowd-sourced repository of images of species from all around the world. In a nutshell, I obtained images of four target species that we taken in Maryland and supplemented this dataset with a subset of all pictures of other species taken in Baltimore. These four target species are:

1. Spotted lanternfly (_Lycorma delicatula_)
2. Mourning dove (_Zenaida macroura_)
3. Red maple (_Acer rubrum_)
4. Horseweed (_Erigeron canadensis_)

We chose these four species because the proposed app was intended to be used at the ESA 2025 conference taking place in Baltimore in August 2025, and these four species could be widely observed in Baltimore around that time of the year.

## How do we determine which images we need?

I used [determine\_inat\_images.py](../sagemaker-run/determine_inat_images.py) to accomplish this task.

Images in the iNaturalist repository are not conveniently organized by location and species name so in order to determine which images we need, I used two large files with observation metadata (observations.csv.gz) and photo metadata (photos.csv.gz) and another, relatively smaller, file that maps taxa id to species name (taxa.csv.gz). Those files all live in the bucket s3://inaturalist-open-data/

The observation metadata file has two columns, `latitude` and `longitude` that I used to determine if an observation was recorded in Maryland (for target species) or in Baltimore (for non-target species). Below are the approximate latitudinal and longitudinal boundaries that I used for both. Note that these give approximate boundaries because geographical entitites like Maryland and Baltimore are not neatly box-shaped.

| Geographical entity | Min lat   | Max lat   | Min lon    | Max lon    |
| ------------------- | --------- | --------- | ---------- | ---------- |
| Maryland            | 37.911717 | 39.723043 | -79.487651 | -75.048939 |
| Baltimore           | 39.19     | 39.37     | -76.71     | -76.52     |

Each record in the observation matadata also has an identifier in the column `observation_uuid` . I used latitude and longitude borders for Maryland to collect all Maryland observations from the observations matadata file using the following Python script. Note that I processed the file in chunks as the file was very large to run in memory for my laptop.

```python
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

```

Note that I used the `taxon_id` column in both the observation and taxon metadata to determine which species is associated with each row in the dataset of observations. Once I had all observations in Maryland, along with the species name associated with each observation record, I filtered those down to observations from Baltimore using the latitude and longitude that approximates the location of Baltimore.

One of the target species, _Acer rubrum_, is a tough one to identify as its leaves look very similar to other related species also found in Baltimore area. In order to give the image classifier enough information to be able to tell those apart from our target species, I added all images from these Acer-like species taken in Maryland. Those additional species are:

1. Sugar maple (_Acer saccharum_)
2. Silver maple (_Acer saccharinum_)
3. Norway maple (_Acer platanoides_)

From Maryland, I collected all observation IDs that are associated with one of the target species or one of the Acer-like species. In contrast, from Baltimore, I collected all observation IDs that are not associated with any of the target or Acer-like species. These observations from Baltimore, along with observations involving Acer-like species, are later used as representatives of the "other" class in image classification.\
In a fashion similar to what I did above to determine observations from Maryland, I collected information about images associated with target or Acer-like species by processing the photo metadata in chunks. From each relevant photo record, I collected `photo_id` and `extension` .

```python
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
```

Note that I calculate an `image_name` column for each record here, which is in the format `<photo_id>_medium.<extension>`. In iNaturalist S3 bucket, each `photo_id` appears as a prefix and for each `photo_id`, there are various sizes of the same image available. I made the decision that the medium size version of each image is a good compromise between image quality and file size.

Once I have the information on relevant images from all of Maryland, I do the same thing for relevant images from Baltimore and put the two datasets together.

Finally, a small number images are associated with multiple observations, which happens, for example, if an image has both a tree and a bird on the tree. In order to avoid situations that a given image could be both associated with a target and non-target species and would therefore have conflicting signal, I took only those images that are associated with a single species, using the following part of the script.

```python
combo_subsample = combo_subsample[~combo_subsample.duplicated(subset=['image_name', 'photo_id', 'extension'], keep=False)]
```

The resulting CSV file has the following columns:

* `species_name`
* `image_name`
* `photo_id`
* `extension`

## How do we get the images that we determined that we need?

Model training will take place on AWS Sagemaker and iNaturalist images are also on AWS, so I worked with Anthropic's Claude to develop a quick script that takes the CSV file generated above and copy images that are needed from iNaturalist S3 bucket to my personal S3 bucket.

This script is stored as [migrate\_in\_s3.py](../sagemaker-run/migrate_in_s3.py)
