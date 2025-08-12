# Training data collection

## Overview

Training data for this model comes from [iNaturalist](https://www.inaturalist.org/), which has a crowd-sourced repository of images of species from all around the world. In a nutshell, I obtained images of four target species that we taken in Maryland and supplemented this dataset with a subset of all pictures of other species taken in Baltimore. These four target species are:

1. Spotted lanternfly (_Lycorma delicatula_)
2. Mourning dove (_Zenaida macroura_)
3. Red maple (_Acer rubrum_)
4. Horseweed (_Erigeron canadensis_)

We chose these four species because the proposed app was intended to be used at the ESA 2025 conference taking place in Baltimore in August 2025, and these four species could be widely observed in Baltimore around that time of the year.

## How do we determine which images we need?

Images in the iNaturalist repository are not conveniently organized by location and species name so in order to determine which images we need, I used two large files with observation metadata (observations.csv.gz) and photo metadata (photos.csv.gz) and another, relatively smaller, file that maps taxa id to species name (taxa.csv.gz). Those files all live in the bucket s3://inaturalist-open-data/

The observation metadata file has two columns, `latitude` and `longitude` that I used to determine if an observation was recorded in Maryland (for target species) or in Baltimore (for non-target species). Below are the approximate latitudinal and longitudinal boundaries that I used for both. Note that these give approximate boundaries because geographical entitites like Maryland and Baltimore are not neatly box-shaped.

| Geographical entity | Min lat   | Max lat   | Min lon    | Max lon    |
| ------------------- | --------- | --------- | ---------- | ---------- |
| Maryland            | 37.911717 | 39.723043 | -79.487651 | -75.048939 |
| Baltimore           | 39.19     | 39.37     | -76.71     | -76.52     |

Each record in the observation matadata also has an identifier in the column `observation_uuid` . I used latitude and longitude borders for Maryland to collect all Maryland observations from the observations matadata file using the following Python script. Note that I processed the file in chunks as the file was very large to run in memory for my laptop.

```python
import pandas as pd

chunk_size = 10000
maryland_observations = []
min_lat = 37.911717
max_lat = 39.723043
min_lon = -1.*79.487651
max_lon = -1.*75.048939
for chunk_number, chunk in enumerate(pd.read_csv('./observations.csv', chunksize=chunk_size, sep = '\t')):

    filtered_chunk = chunk.query(f"(latitude <= {max_lat}) & (latitude >= {min_lat}) & (longitude <= {max_lon}) & (longitude >= {min_lon})")

    if not filtered_chunk.empty:
        print(f"Appending {len(filtered_chunk)} records")
        maryland_observations.append(filtered_chunk)
        
maryland_obs_df = pd.concat(maryland_observations, ignore_index=True)
```

&#x20;  &#x20;
