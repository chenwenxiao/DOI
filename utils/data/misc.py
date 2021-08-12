from typing import *

import numpy as np
from terminaltables import AsciiTable

__all__ = ['format_labels_grid']


def format_labels_grid(labels: Sequence[str],
                       n_cols: Optional[int] = None) -> str:
    labels = list(labels)
    if not labels:
        raise ValueError(f'`labels` must not be empty.')
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(len(labels))))
    n_rows = (len(labels) + n_cols - 1) // n_cols

    tbl = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < len(labels):
                row.append(labels[idx])
            else:
                row.append('')
        tbl.append(row)

    tbl = AsciiTable(tbl)
    tbl.inner_heading_row_border = 0
    return tbl.table
