# Input data

| File | Content | Used by |
|---|---|---|
| `DEM_LaPalma_COP30.tif` | Copernicus DEM GLO-30 (DSM), La Palma mosaic, EPSG:4326, float32 m, ~2.3 MB | Default DEM (`--dem-source copernicus`) |
| `DEM_LaPalma.tif` | Legacy DEM. Source not recorded; likely a 1-arcsec global DEM reprojected to EPSG:3857 (int16 m, nodata −32768) | `--dem-source legacy` |
| `Emission_Pt.gpkg` | Vent points: layer `2836460n_1751580w` (Tajogaite cone, default) and layer `emision_pto` (earlier default, ~2.9 km ENE) | `--emission-layer` |
| `DEM_gaussian.tif` | Small synthetic DEM; not used by the script (gaussian mode builds its own) | — |
| `Erupcion-Volcan-Cumbre-Vieja-La-Palma-Landsat8-26-09-21.jpg`, `output.png` | Images used by the legacy notebook | `notebooks/legacy/` |

## Copernicus GLO-30 DEM

`DEM_LaPalma_COP30.tif` is small enough (~2.3 MB) to be versioned. To rebuild it from
the source tiles, run this from the repository root:

```bash
python scripts/download_dem.py               # writes data/DEM_LaPalma_COP30.tif
python scripts/download_dem.py --cache-dir ~/cop30_tiles   # keep the raw tiles
```

The script works as follows:

1. It downloads `Copernicus_DSM_COG_10_N28_00_W018_00_DEM.tif` over HTTPS from
   `https://copernicus-dem-30m.s3.amazonaws.com/`. If HTTP fails, it falls back to
   `aws s3 cp --no-sign-request`, which needs the AWS CLI.
2. It checks whether La Palma's land reaches the tile's western edge (18° W). The
   island reaches ~18.007° W, so the script also downloads
   `Copernicus_DSM_COG_10_N28_00_W019_00_DEM.tif` and mosaics the two tiles.
3. It crops the mosaic to the island's land extent plus a 0.02° margin. It fails if any
   land touches the crop border, i.e. if the island is incomplete.
4. It writes a DEFLATE-compressed GeoTIFF whose tags record the product, the tiles,
   the processing, the access date and the attribution.

The resulting raster is 1603 × 1162 pixels, covering 18.027–17.704° W and
28.433–28.878° N. The file in the repository was built on 2026-11-17. A rebuild on a
different date should give the same pixel values unless the source tiles change.

**Licence and attribution.** Copernicus DEM GLO-30 Public is available on a free basis
under the ESA licence linked from the
[COP-DEM collection page](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
Any use of this file must carry the following notice:

> produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and
> Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all
> rights reserved
