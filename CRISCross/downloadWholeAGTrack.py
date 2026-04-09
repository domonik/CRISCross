
from alphagenome.models import dna_client
from alphagenome.data.genome import Interval
from alphagenome.models.dna_client import OutputType

import pandas as pd
import time
import os
import numpy as np



CHROMOSOME_SIZES = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415
}





#WINDOW_SIZE = 2048
#WINDOW_SIZE = 2**17
WINDOW = 2**20

EXTRACT = 2**16         # 16,384
FLANK = (WINDOW - EXTRACT) // 2



import grpc

def run_batch_prediction(dna_model, intervals, organism, ontology_terms, OutputType,
                         max_retries=5, base_delay=1, max_delay=60):
    attempt = 0

    while attempt < max_retries:
        try:
            outputs = dna_model.predict_intervals(
                intervals=intervals,
                requested_outputs=set(OutputType),
                ontology_terms=ontology_terms,
                organism=organism,
                progress_bar=True,
                max_workers=5
            )
            return outputs

        except grpc._channel._MultiThreadedRendezvous as e:
            # Check if the error is RESOURCE_EXHAUSTED (quota exceeded)
            attempt += 1
            wait_time = min(base_delay * (2 ** (attempt - 1)), max_delay)
            print(f"Quota exceeded. Retrying in {wait_time} seconds (attempt {attempt}/{max_retries})...")
            print(e)
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error during batch prediction: {e}")
            attempt += 1
            wait_time = min(base_delay * (2 ** (attempt - 1)), max_delay)
            print(f"Quota exceeded. Retrying in {wait_time} seconds (attempt {attempt}/{max_retries})...")
            print(e)

            time.sleep(wait_time)
            return []

    print("Max retries exceeded. Returning empty batch.")
    return []

#### change ontology depending on cell type, Tcell = 'CL:0000084' ######
def extract_batch_features(rows_df, ontology_terms=['CL:0000624'], skipped_rows_path="./alphagenome_testcode/skipped_intervals.csv"):
    intervals = []
    valid_rows = []
    skipped_rows = []

    for _, row in rows_df.iterrows():
        chrom = row['chr']
        start = row['start']

        # Normalize chromosome name
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom

        # Update row if we modified chrom
        row = row.copy()
        row['chr'] = chrom

        intervals.append(create_interval_from_row(row))
        valid_rows.append(row)

    if not intervals:
        return []

    batch_features = run_batch_prediction(dna_model=dna_model, intervals=intervals,
                                          ontology_terms=ontology_terms, OutputType=OutputType,
                                          max_retries=25, base_delay=5, max_delay=60)
    

    

    return batch_features



def create_interval_from_row(row):       
    return Interval(
        chromosome=row['chrom'],
        start=row["start"],
        end=row["end"],
    )


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ontology",
        required=True,
        help="Ontology CURIE (e.g. CL:0000624)"
    )
    parser.add_argument(
        "--organism",
        default="HOMO_SAPIENS",
        choices=["HOMO_SAPIENS", "MUS_MUSCULUS"],
        help="Organism for AlphaGenome (default: HOMO_SAPIENS)"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["ATAC", "DNASE", "CHIP_HISTONE", "RNA_SEQ"],
        help="Features to extract (default: ATAC DNASE CHIP_HISTONE RNA_SEQ)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: AGTensors{ONT})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without making API calls"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test: only download chr1 with a single batch"
    )
    return parser.parse_args()


def build_track_metadata(organism: str, ontology: str, features: list):
    """
    Build track metadata from AlphaGenome API, replacing external CSV dependency.

    Args:
        organism: Organism name (HOMO_SAPIENS, MUS_MUSCULUS)
        ontology: Ontology CURIE to filter tracks
        features: List of feature types to include

    Returns:
        DataFrame with track metadata including: name, output_type, newTrackIndex,
        biosample_name, and other relevant columns
    """
    organism_enum = getattr(dna_client.Organism, organism)
    dna_model = dna_client.create(os.environ['AGAPIKEY'])
    meta = dna_model.output_metadata(organism=organism_enum)

    # Combine all track types with output_type indicator
    combined_dfs = []
    track_type_map = {
        'atac': 'ATAC',
        'dnase': 'DNASE',
        'chip_histone': 'CHIP_HISTONE',
        'rna_seq': 'RNA_SEQ'
    }

    for attr_name, output_type in track_type_map.items():
        if output_type not in features:
            continue
        df = getattr(meta, attr_name).copy()
        df['output_type'] = output_type
        combined_dfs.append(df)

    if not combined_dfs:
        raise ValueError(f"No features matched: {features}")

    combined = pd.concat(combined_dfs, ignore_index=True)

    # Filter by ontology
    combined = combined[combined["ontology_curie"] == ontology]

    if combined.empty:
        raise ValueError(f"No tracks found for ontology {ontology} with features {features}")

    # Create newTrackIndex for unique track identification
    if "CHIP_HISTONE" in features:
        combined.loc[combined["output_type"] == "CHIP_HISTONE", "newTrackIndex"] = \
            combined.loc[combined["output_type"] == "CHIP_HISTONE", "histone_mark"]
    if "RNA_SEQ" in features:
        combined.loc[combined["output_type"] == "RNA_SEQ", "newTrackIndex"] = \
            combined.loc[combined["output_type"] == "RNA_SEQ", "strand"] + "_" + \
            combined.loc[combined["output_type"] == "RNA_SEQ", "Assay title"]
    if "ATAC" in features or "DNASE" in features:
        combined.loc[combined["output_type"].isin(["ATAC", "DNASE"]), "newTrackIndex"] = \
            combined.loc[combined["output_type"].isin(["ATAC", "DNASE"]), "name"]

    return combined


def main():
    BATCH_SIZE = 100

    print("Extracting features for all intervals...")
    args = parse_args()
    ONT = args.ontology
    FEATURES = args.features
    ORGANISM = args.organism

    # Set output directory
    OUTDIR = args.output_dir or f"AGTensors{ONT}"

    # Build track metadata from AlphaGenome API (no external CSV needed)
    print(f"Fetching track metadata for ontology {ONT} and features {FEATURES}...")
    organism_enum = getattr(dna_client.Organism, ORGANISM)
    ag_info = build_track_metadata(organism=ORGANISM, ontology=ONT, features=FEATURES)
    print(f"Found {len(ag_info)} tracks")

    # Build TRACK_NAMES and NAMETRACKNAMMPING from ag_info
    TRACK_NAMES = ag_info.groupby("output_type")["newTrackIndex"].unique().reset_index()
    NAMETRACKNAMMPING = {row["name"]: row["newTrackIndex"] for _, row in ag_info.iterrows()}
    TRACK_NAMES = TRACK_NAMES.explode("newTrackIndex", ignore_index=True)
    TRACK_NAMES["list_index"] = TRACK_NAMES.groupby(["output_type"]).cumcount()

    # Prepare interval rows for all chromosomes
    rows = []
    # Filter to chr1 only for quick test
    chromosomes_to_process = {"chr1": CHROMOSOME_SIZES["chr1"]} if args.quick_test else CHROMOSOME_SIZES

    for chrom, chrom_size in chromosomes_to_process.items():
        extract_start = 0
        batch_count = 0
        while extract_start < chrom_size:
            extract_end = min(extract_start + EXTRACT, chrom_size)
            start = extract_start - FLANK
            end = start + WINDOW

            if start < 0:
                start = 0
                end = min(WINDOW, chrom_size)

            if end > chrom_size:
                end = chrom_size
                start = max(0, chrom_size - WINDOW)

            rows.append({
                "chrom": chrom,
                "start": start,
                "end": end,
                "extract_start": extract_start,
                "extract_end": extract_end
            })
            extract_start += EXTRACT
            batch_count += 1
            # For quick test, only process the first batch
            if args.quick_test and batch_count >= 1:
                break
        # For quick test, stop after first chromosome's first batch
        if args.quick_test:
            break

    df = pd.DataFrame(rows)
    df = df.groupby(["chrom", "start", "end"], as_index=False).agg({"extract_start": "min", "extract_end": "max"})

    # Dry run: show what would happen without making API calls
    if args.dry_run:
        print("\n" + "="*50)
        print("DRY RUN - No API calls will be made")
        print("="*50)
        print(f"Output directory: {OUTDIR}")
        print(f"Organism: {ORGANISM}")
        print(f"Features: {FEATURES}")
        print(f"Tracks to download: {len(TRACK_NAMES)}")
        print(f"Intervals to process: {len(df)}")
        print("\nExpected output files (sample):")
        for track in TRACK_NAMES['newTrackIndex'][:5]:
            expected_size = CHROMOSOME_SIZES.get('chr1', 0)
            print(f"  - {track}_chr1.npy (shape: ({expected_size}, 1), ~{expected_size*4/1e6:.1f}MB)")
        print("\nChromosomes to process:")
        for chrom in df['chrom'].unique()[:5]:
            chrom_intervals = len(df[df['chrom'] == chrom])
            print(f"  - {chrom}: {chrom_intervals} batches")
        if len(df['chrom'].unique()) > 5:
            print(f"  ... and {len(df['chrom'].unique()) - 5} more chromosomes")
        exit(0)

    apikey = os.environ['AGAPIKEY']
    dna_model = dna_client.create(apikey)

    if not os.path.isdir(OUTDIR):
        os.mkdir(OUTDIR)



    def create_arrays(outputs):
        output = outputs[0]
        sizes = {}
        for idx, row in TRACK_NAMES.iterrows():
            for chrom, chrom_size in CHROMOSOME_SIZES.items():
                track_name = get_track_name(row)
                filename = os.path.join(OUTDIR, f"{track_name}_{chrom}.npy")
                data = output.get(OutputType[row["output_type"]])
                assert data is not None
                shape = chrom_size, 1
                mmap_array = np.memmap(filename, dtype=np.float32, mode="w+", shape=shape)
                mmap_array.flush()


    def get_track_name(track_row):
        """Extract clean track name for output files."""
        output_type = track_row["output_type"]
        newTrackIndex = track_row["newTrackIndex"]

        # For ATAC/DNASE, use the feature name directly instead of the full AlphaGenome track name
        if output_type == "ATAC":
            return "ATAC"
        if output_type == "DNASE":
            return "DNASE"

        # For other types, strip ontology prefix
        if ":" in newTrackIndex:
            name = newTrackIndex.split(":", 1)[1].strip()
            name = name.lstrip("0123456789 ").strip()
            return name

        return newTrackIndex

    def fill_array(outputs, rows):
        assert len(outputs) == len(rows)


        for idx, track_row in TRACK_NAMES.iterrows():
            fdata = outputs[0].get(OutputType[track_row["output_type"]])
            chrom = rows.iloc[0]["chrom"]
            track_name = get_track_name(track_row)
            filename = os.path.join(OUTDIR, f"{track_name}_{chrom}.npy")
            shape = CHROMOSOME_SIZES[chrom], 1

            mmap_array = np.memmap(filename, dtype=np.float32, mode='r+', shape=shape)
            for i, output in enumerate(outputs):
                row = rows.iloc[i]
                data = output.get(OutputType[track_row["output_type"]])
                row_idx = fdata.metadata["name"].map(NAMETRACKNAMMPING)
                if data.resolution != 1:
                    data = data.change_resolution(resolution=1)
                if not pd.isna(row_idx).any():
                    if track_row["list_index"] >= data.values.shape[1]:
                        breakpoint()
                    vals = data.values[:, track_row["list_index"]][..., None]
                else:
                    vals = data.values
                vals_offset_start = row["extract_start"] - row["start"]
                vals_offset_end   = row["extract_end"] - row["start"]
                mmap_array[row["extract_start"]:row["extract_end"]] = vals[vals_offset_start:vals_offset_end]
            mmap_array.flush()





    idx = 0
    df["Intervals"] = df.apply(create_interval_from_row, axis=1)

    # Quick test summary
    if args.quick_test:
        total_batches = sum(len(group) // BATCH_SIZE + 1 for _, group in df.groupby(["chrom"]))
        print(f"\nQUICK TEST MODE: Downloading {len(TRACK_NAMES)} tracks on chr1 ({len(df)} intervals = {total_batches} batch)")

    for (chrom), group in df.groupby(["chrom"]):
        print("Extracting features in batches...")

        for batch_idx, start_idx in enumerate(range(0, len(group), BATCH_SIZE)):
            end_idx = min(start_idx + BATCH_SIZE, len(group))
            batch_df = group.iloc[start_idx:end_idx]

            print(f"Processing batch {start_idx} to {end_idx}...")
            batch_output = run_batch_prediction(dna_model=dna_model, organism=organism_enum, intervals=batch_df["Intervals"],
                                            ontology_terms=[ONT], OutputType=OutputType, max_retries=25, base_delay=5, max_delay=60)
        
            if idx == 0:
                sizes = create_arrays(batch_output)
                idx += 1
            fill_array(outputs=batch_output, rows=batch_df)


if __name__ == "__main__":
    main()







