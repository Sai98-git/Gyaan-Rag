# MSMARCO-XI Dataset Repository Report

This report details the structure, contents, schemas, and ingestion analysis of the `ai4bharat/MSMARCO-XI` dataset.

---

## 1. Repository Structure & Files

The Hugging Face repository `ai4bharat/MSMARCO-XI` contains 30 files in total:
* **3 Configuration/Metadata files**: `.gitattributes`, `README.md`, and `ms_marco_translations.py` (custom dataset builder config).
* **27 Parquet data files**: Distributed across `train/` (13 files) and `validation/` (14 files).

### File List and Sizes
All file sizes were queried via the Hugging Face Hub API (`HfApi`):

| Path | Size (Bytes) | Size (GB) | Split | Language (ISO-639-3) |
| :--- | :--- | :--- | :--- | :--- |
| `train/asmtrain.parquet` | 3,789,324,486 | 3.5291 GB | Train | Assamese (`asm`) |
| `train/bentrain.parquet` | 3,727,370,553 | 3.4714 GB | Train | Bengali (`ben`) |
| `train/gujtrain.parquet` | 3,718,439,546 | 3.4631 GB | Train | Gujarati (`guj`) |
| `train/hintrain.parquet` | 3,719,813,179 | 3.4643 GB | Train | Hindi (`hin`) |
| `train/kantrain.parquet` | 3,892,610,876 | 3.6253 GB | Train | Kannada (`kan`) |
| `train/maltrain.parquet` | 3,986,867,780 | 3.7131 GB | Train | Malayalam (`mal`) |
| `train/martrain.parquet` | 3,756,780,048 | 3.4988 GB | Train | Marathi (`mar`) |
| `train/neptrain.parquet` | 3,633,169,552 | 3.3837 GB | Train | Nepali (`nep`) |
| `train/oritrain.parquet` | 3,778,629,786 | 3.5191 GB | Train | Odia (`ori`) |
| `train/pantrain.parquet` | 3,708,783,621 | 3.4541 GB | Train | Punjabi (`pan`) |
| `train/santrain.parquet` | 4,001,524,369 | 3.7267 GB | Train | Sanskrit (`san`) |
| `train/tamtrain.parquet` | 3,987,720,629 | 3.7139 GB | Train | Tamil (`tam`) |
| `train/urdtrain.parquet` | 3,339,235,620 | 3.1099 GB | Train | Urdu (`urd`) |
| `validation/asmval.parquet` | 470,284,966 | 0.4380 GB | Validation | Assamese (`asm`) |
| `validation/benval.parquet` | 462,777,608 | 0.4310 GB | Validation | Bengali (`ben`) |
| `validation/gujval.parquet` | 461,249,303 | 0.4296 GB | Validation | Gujarati (`guj`) |
| `validation/hinval.parquet` | 461,888,616 | 0.4302 GB | Validation | Hindi (`hin`) |
| `validation/kanval.parquet` | 482,734,885 | 0.4496 GB | Validation | Kannada (`kan`) |
| `validation/malval.parquet` | 493,598,017 | 0.4597 GB | Validation | Malayalam (`mal`) |
| `validation/marval.parquet` | 473,618,819 | 0.4411 GB | Validation | Marathi (`mar`) |
| `validation/nepval.parquet` | 466,513,930 | 0.4345 GB | Validation | Nepali (`nep`) |
| `validation/orival.parquet` | 466,515,731 | 0.4345 GB | Validation | Odia (`ori`) |
| `validation/panval.parquet` | 459,536,147 | 0.4280 GB | Validation | Punjabi (`pan`) |
| `validation/sanval.parquet` | 494,213,431 | 0.4603 GB | Validation | Sanskrit (`san`) |
| `validation/tamval.parquet` | 493,049,000 | 0.4592 GB | Validation | Tamil (`tam`) |
| `validation/telval.parquet` | 474,142,748 | 0.4416 GB | Validation | Telugu (`tel`) |
| `validation/urdval.parquet` | 419,206,311 | 0.3904 GB | Validation | Urdu (`urd`) |

> [IMPORTANT]
> **Telugu (`tel`) is missing a training split** file (`train/teltrain.parquet`) in the repository. It only has a validation split file (`validation/telval.parquet`).

---

## 2. Parquet Schema & Statistics

The schema was determined by querying the Parquet file metadata footer directly using `pyarrow.parquet.ParquetFile` and `fsspec` HTTP range requests. No files were fully downloaded.

### Command/Tool Used
```python
import pyarrow.parquet as pq
import fsspec
url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/hintrain.parquet"
f = fsspec.open(url).open()
pf = pq.ParquetFile(f)
schema = pf.schema
metadata = pf.metadata
```

### Representative Schema
The schema contains **17 columns** in total:

```text
required group field_id=-1 schema {
  optional binary field_id=-1 source_lang (String);
  optional binary field_id=-1 target_lang (String);
  optional group field_id=-1 meta {
    optional int64 field_id=-1 frequency_penalty;
    optional int64 field_id=-1 max_tokens;
    optional binary field_id=-1 model_name (String);
    optional int64 field_id=-1 presence_penalty;
    optional int64 field_id=-1 temperature;
    optional int64 field_id=-1 top_p;
  }
  optional binary field_id=-1 Answer (String);
  optional int64 field_id=-1 query_id;
  optional binary field_id=-1 query_type (String);
  optional group field_id=-1 passages {
    optional group field_id=-1 English_passages (List) {
      repeated group field_id=-1 list {
        optional binary field_id=-1 element (String);
      }
    }
    optional group field_id=-1 Translated_passages (List) {
      repeated group field_id=-1 list {
        optional binary field_id=-1 element (String);
      }
    }
    optional group field_id=-1 is_selected (List) {
      repeated group field_id=-1 list {
        optional int64 field_id=-1 element;
      }
    }
  }
  optional binary field_id=-1 Eng_Query (String);
  optional binary field_id=-1 Eng_Answer (String);
  optional binary field_id=-1 query (String);
}
```

### Key Statistical Properties
* **Row Group Counts**: **Every single Parquet file has exactly 1 row group** (`num_row_groups = 1`).
* **Row Counts**:
  * **Train Splits**: Most languages have exactly `778,638` rows, but Marathi has `765,873`, Nepali has `754,154`, Odia has `782,282`, and Urdu has `770,089` rows.
  * **Validation Splits**: Every validation file contains exactly `97,941` rows.

> [WARNING]
> Because each file contains only **1 row group**, PyArrow is forced to read column data for the entire row group to fetch even a single row. This means any remote stream operation must download gigabytes of column data, explaining why streaming over HTTP times out or hangs.

---

## 3. Ingestion & Development Strategy

### Recommended Development Corpus (`CORPUS_MODE=sample`)
* **File Target**: Use the **validation split of the default language** (`validation/hinval.parquet`, Hindi).
* **Rationale**:
  1. The validation files are **~10x smaller** than training files (~430 MB vs ~3.5 GB) and can be downloaded in ~2 minutes at 2.4 MB/s.
  2. The validation splits have the exact same schema and structure as the training splits.
  3. `hinval.parquet` provides up to `97,941` records, which is more than sufficient for the target development corpus of `10,000–50,000` records (e.g. default `MAX_RECORDS=25000`).

### Configurable Ingestion Modes
The system config (`backend/core/config.py`) will support:
1. `CORPUS_MODE=sample`: Downloads/caches `validation/hinval.parquet` (or the validation file of the chosen `DATA_LANGUAGE`) and processes up to `MAX_RECORDS` (e.g., 25,000).
2. `CORPUS_MODE=language`: Downloads/caches the full training split file (e.g., `train/hintrain.parquet`, 3.46 GB) for the language specified in `DATA_LANGUAGE` and loads all records.
3. `CORPUS_MODE=full`: Not implemented in this phase.

### Ingestion Method
1. Use `huggingface_hub.hf_hub_download` to download the specific target Parquet file to the local cache.
2. Read the local cached file using PyArrow / pandas in a memory-efficient streamed row generator or chunked reader (e.g. using `pyarrow.parquet.read_table` or PyArrow datasets API).
3. Map columns to our normalized internal schema.

---

## 4. Expected Storage & Limitations

### Storage Requirements
* **Sample Mode**: ~430 MB (disk cache).
* **Language Mode (Hindi)**: ~3.46 GB.
* **Full Ingestion**: ~55 GB (all parquet files cached).

### Dataset Limitations
1. **No Telugu Train Split**: Telugu (`tel`) cannot be ingested in `train` split due to missing repository files.
2. **Broken HF Loader**: The `load_dataset("ai4bharat/MSMARCO-XI")` call is completely unusable because it executes a broken script mapping to non-existent JSONL files.
3. **No Parquet Shard Partitioning**: Since there is only 1 row group per file, incremental remote reading is not possible; the file must be downloaded locally first.
