# Palmer Penguins data dictionary

One row describes one recorded penguin observation. The simplified Kaggle file
does not provide a unique individual identifier. Its sample covers three species
from three islands in the Palmer Archipelago. Counts are not population estimates.

| Raw column | Prepared column | Meaning/unit | Handling |
| --- | --- | --- | --- |
| `species` | `species` | Adelie, Chinstrap, Gentoo | Required group label |
| `island` | `island` | Biscoe, Dream, Torgersen | Required location label |
| `culmen_length_mm` | `bill_length_mm` | Bill length, millimetres | Positive numeric; preserve missing |
| `culmen_depth_mm` | `bill_depth_mm` | Bill depth, millimetres | Positive numeric; preserve missing |
| `flipper_length_mm` | `flipper_length_mm` | Flipper length, millimetres | Positive numeric; preserve missing |
| `body_mass_g` | `body_mass_g` | Body mass, grams | Positive numeric; preserve missing |
| `sex` | `sex` | Recorded male/female label | Lowercase; `.`/NA becomes missing |
| Not in raw file | `source_row` | One-based source record number | Traceability only, not an animal ID |

Pandas parses CSV `NA` markers as missing. A separate normalization step handles
the literal `.` category. No mean/median imputation is used for descriptive EDA.
Complete-case analysis is a sensitivity check, not the default for every chart.

Source and licensing: [DATA_SOURCES.md](../../docs/DATA_SOURCES.md).
