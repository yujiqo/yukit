import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt


RawCell = str
RawRow = dict[str, RawCell]
RawData = list[RawRow]

Cell = str | float | None
Row = dict[str, Cell]
Data = list[Row]

DEFAULT_NONE_VALUES: frozenset[str] = frozenset(
    {
        "",
        "na",
        "n/a",
        "nan",
        "nil",
        "null",
        "none",
        "unknown",
        "undefined",
        "missing",
        "blank",
    }
)


class Dataset:
    data: Data
    columns: list[str]
    seed: int | None
    none_values: frozenset[str]

    def __init__(
        self,
        data: RawData,
        columns: list[str],
        none_values: Iterable[str] | None = None,
        seed: int | None = None,
    ):
        normalized_none_values = DEFAULT_NONE_VALUES
        if none_values is not None:
            normalized_none_values = frozenset(
                value.strip().lower() for value in none_values
            )

        self.seed = seed
        self.none_values = normalized_none_values
        self.columns = list(columns)
        self.data = self._parse(data, self.columns, normalized_none_values)

    @staticmethod
    def _parse(data: RawData, columns: list[str], none_values: frozenset[str]) -> Data:
        parsed_data: Data = [{} for _ in data]

        for column in columns:
            cleaned_values: list[Cell] = []
            for row in data:
                if column not in row:
                    raise ValueError(f'The column "{column}" is missing from a row!')

                value = row[column].strip()
                cleaned_values.append(None if value.lower() in none_values else value)

            try:
                column_values: list[Cell] = [
                    None if value is None else float(value) for value in cleaned_values
                ]
            except ValueError:
                column_values = cleaned_values

            for parsed_row, value in zip(parsed_data, column_values):
                parsed_row[column] = value

        return parsed_data

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        if not path.exists():
            raise FileNotFoundError(f"Dataset in the {path.resolve()} doesn't exist!")

        with path.open(newline="") as file:
            lines = csv.reader(file, skipinitialspace=True)

            try:
                header = next(lines)
            except StopIteration:
                raise ValueError(f"Dataset in the {path.resolve()} is empty!") from None

            columns = [column.strip() for column in header]
            if not columns:
                raise ValueError(f"Dataset in the {path.resolve()} has no columns!")
            if len(set(columns)) != len(columns):
                raise ValueError("Dataset contains duplicate column names!")

            data: RawData = []
            for line_number, line in enumerate(lines, start=2):
                if len(line) != len(columns):
                    raise ValueError(
                        f"Row {line_number} has {len(line)} values, but {len(columns)} were expected!"
                    )
                data.append(dict(zip(columns, line)))

        return cls(data, columns)

    def head(self, n: int = 5) -> None:
        max_lengths: dict[str, int] = {}
        for column in self.columns:
            max_length = 0
            for row in self.data:
                cell_length = len(str(row[column]))
                if max_length < cell_length:
                    max_length = cell_length
            max_lengths[column] = max_length

        print_formats: list[str] = []
        header = ""
        for column in self.columns:
            print_format = "{" + f":<{len(column) + max_lengths[column]}" + "}"
            print_formats.append(print_format)
            header += print_format.format(column)

        print(header)
        print("-" * len(header))

        n = max(0, min(n, len(self.data)))
        for i in range(n):
            row = ""
            for j, column in enumerate(self.columns):
                row += print_formats[j].format(str(self.data[i][column]))
            print(row)

    def get(self, index: int) -> Row:
        if index < 0 or index >= len(self.data):
            raise IndexError("Index is out of dataset's range!")
        return self.data[index]

    def add(self, data: Row | list[Row], axis: Literal["row", "col"]) -> None:
        if axis == "row":
            if not isinstance(data, dict):
                raise TypeError("data is not dict, while axis is set to row!")
            if set(data) != set(self.columns):
                raise ValueError(
                    "Row keys must match the dataset columns, while axis is set to row!"
                )
            self.data.append(data)

        elif axis == "col":
            if not isinstance(data, list):
                raise TypeError("data is not list, while axis is set to col!")
            if not data:
                raise ValueError("data is empty, while axis is set to col!")
            if len(data) != len(self.data):
                raise ValueError(
                    "data needs to be same length as dataset, while axis set to col!"
                )

            keys: set[str] = set()
            for row in data:
                if not isinstance(row, dict) or len(row.keys()) != 1:
                    raise ValueError(
                        "Each row of data should contain only single column, while axis set to col!"
                    )
                keys.update(row.keys())

            if len(keys) != 1:
                raise ValueError(
                    "All rows of data should contain the same column, while axis set to col!"
                )

            key = next(iter(keys))
            if key in self.columns:
                raise ValueError(f'The column "{key}" already exists in the dataset!')

            for dataset_row, column_row in zip(self.data, data):
                dataset_row[key] = column_row[key]
            self.columns.append(key)

        else:
            raise ValueError(f'Invalid axis "{axis}", expected "row" or "col"!')

    def drop(self, data: list[int] | list[str], axis: Literal["row", "col"]) -> None:
        if axis == "row":
            indexes: list[int] = []
            for index in data:
                if not isinstance(index, int):
                    raise TypeError(
                        "The data should contain only numbers, while axis is row!"
                    )
                if index < 0 or index >= len(self.data):
                    raise IndexError(f'Index "{index}" is out of dataset\'s range!')
                indexes.append(index)

            dropped_indexes = set(indexes)
            self.data = [
                row for i, row in enumerate(self.data) if i not in dropped_indexes
            ]

        elif axis == "col":
            columns: list[str] = []
            for column in data:
                if not isinstance(column, str):
                    raise TypeError(
                        "The data should contain only strings, while axis is col!"
                    )
                self._is_column_exists(column)
                columns.append(column)

            self.columns = [column for column in self.columns if column not in columns]
            for row in self.data:
                for column in columns:
                    del row[column]

        else:
            raise ValueError(f'Invalid axis "{axis}", expected "row" or "col"!')

    def unique(self, column: str) -> set[Cell]:
        self._is_column_exists(column)

        return {row[column] for row in self.data}

    def empty(self, column: str) -> list[int]:
        self._is_column_exists(column)

        return [i for i, row in enumerate(self.data) if row[column] is None]

    def duplicates(self, mode: Literal["exact", "approximate"]) -> list[int]:
        duplicates: list[int] = []

        if mode == "exact":
            seen: set[tuple[Cell, ...]] = set()

            for i, row in enumerate(self.data):
                key = tuple(row[column] for column in self.columns)

                if key in seen:
                    duplicates.append(i)
                else:
                    seen.add(key)

        elif mode == "approximate":
            raise NotImplementedError(
                "Approximate duplicate detection is not implemented yet!"
            )

        else:
            raise ValueError(
                f'Invalid mode "{mode}", expected "exact" or "approximate"!'
            )

        return duplicates

    def distribution(
        self, column: str, kind: Literal["categorical", "numerical"], bins: int = 10
    ) -> None:
        self._is_column_exists(column)

        values: list[Cell] = [row[column] for row in self.data]

        if kind == "categorical":
            counts = Counter(values)
            plt.bar([str(column) for column in counts], list(counts.values()))
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Count")

        elif kind == "numerical":
            numbers: list[float] = []
            for value in values:
                if value is None:
                    continue

                try:
                    numbers.append(float(value))
                except (TypeError, ValueError):
                    raise ValueError(
                        f'The column "{column}" contains non-numeric values!'
                    )

            if not numbers:
                raise ValueError(f'The column "{column}" has no numeric values!')

            plt.hist(numbers, bins=bins, edgecolor="white")
            plt.ylabel("Frequency")

        else:
            raise ValueError(
                f'Invalid kind "{kind}", expected "categorical" or "numerical"!'
            )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.tight_layout()
        plt.show()

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def _is_column_exists(self, column: str) -> None:
        if column not in self.columns:
            raise ValueError(f'The column "{column}" is not in the dataset!')
