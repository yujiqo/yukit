import csv
import math
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .exceptions import (
    ManualOperationError,
    MismatchError,
    PhaseMismatchError,
    StepMismatchError,
)


RawCell = str
RawRow = dict[str, RawCell]
RawData = list[RawRow]

Cell = str | int | float | None
Row = dict[str, Cell]
Data = list[Row]

ColumnRole = Literal["numerical", "ordinal", "nominal", "text", "target"]

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
    phase: Literal["manual", "automatic"]
    step: int
    numerical: list[str]
    ordinal: list[str]
    nominal: list[str]
    text: list[str]
    target: list[str]
    ordinal_order: dict[str, list[str]]
    train_indices: list[int]
    val_indices: list[int] | None
    test_indices: list[int]
    fill_values: dict[str, Cell]
    scale_params: dict[str, tuple[float, float]]
    target_mappings: dict[str, dict[str, int]]
    _manual_snapshot: tuple[Data, list[str]] | None

    def __init__(
        self,
        data: RawData,
        columns: list[str],
        none_values: Iterable[str] | None = None,
        seed: int | None = None,
    ):
        normalized_none_values = DEFAULT_NONE_VALUES
        if none_values is not None:
            normalized_none_values = self._normalize_none_values(none_values)

        self.seed = seed
        self.none_values = normalized_none_values
        self.columns = list(columns)
        self.data = self._parse(data, self.columns, normalized_none_values)

        self.phase = "manual"
        self.step = 0
        self.numerical = []
        self.ordinal = []
        self.nominal = []
        self.text = []
        self.target = []
        self.ordinal_order = {}
        self.train_indices = []
        self.val_indices = None
        self.test_indices = []
        self.fill_values = {}
        self.scale_params = {}
        self.target_mappings = {}
        self._manual_snapshot = None

    @staticmethod
    def _normalize_none_values(none_values: Iterable[str]) -> frozenset[str]:
        return frozenset(value.strip().lower() for value in none_values)

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

    @staticmethod
    def _copy_data(data: Data) -> Data:
        return [dict(row) for row in data]

    @staticmethod
    def _category_key(value: Cell) -> str:
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else str(value)
        return str(value)

    def _check_phase(self, phase: Literal["manual", "automatic"]) -> bool:
        return self.phase == phase

    def _check_step(self, step: int) -> bool:
        return self.step == step - 1

    def _check_write_method_possibility(self) -> None:
        if not self._check_phase("manual"):
            raise ManualOperationError(
                "Write operations are blocked during automatic phase!"
            )

    def _check_automatic_step_method_possibility(self, step: int) -> None:
        if not self._check_phase("automatic"):
            raise PhaseMismatchError(
                "Automatic methods are allowed only in automatic phase!"
            )
        if not self._check_step(step):
            raise StepMismatchError("Current step requires previous step to be done!")

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
        self._print_head(self.data, self.columns, n)

    @staticmethod
    def _print_head(data: Data, columns: list[str], n: int = 5) -> None:
        max_lengths: dict[str, int] = {}
        for column in columns:
            max_length = 0
            for row in data:
                cell_length = len(str(row[column]))
                if max_length < cell_length:
                    max_length = cell_length
            max_lengths[column] = max_length

        print_formats: list[str] = []
        header = ""
        for column in columns:
            print_format = "{" + f":<{len(column) + max_lengths[column] + 2}" + "}"
            print_formats.append(print_format)
            header += print_format.format(column)

        print(header)
        print("-" * len(header))

        n = max(0, min(n, len(data)))
        for i in range(n):
            row = ""
            for j, column in enumerate(columns):
                row += print_formats[j].format(str(data[i][column]))
            print(row)

    def get(self, index: int) -> Row:
        if index < 0 or index >= len(self.data):
            raise IndexError("Index is out of dataset's range!")
        return self.data[index]

    def add(self, data: Row | list[Row], axis: Literal["row", "col"]) -> None:
        self._check_write_method_possibility()

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
        self._check_write_method_possibility()

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
            plt.bar([str(value) for value in counts], list(counts.values()))
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

    def _is_column_exists(self, column: str) -> None:
        if column not in self.columns:
            raise ValueError(f'The column "{column}" is not in the dataset!')

    def _column_role(self, column: str) -> ColumnRole | None:
        if column in self.numerical:
            return "numerical"
        if column in self.ordinal:
            return "ordinal"
        if column in self.nominal:
            return "nominal"
        if column in self.text:
            return "text"
        if column in self.target:
            return "target"
        return None

    def _rows(self, indices: list[int]) -> Data:
        return [self.data[index] for index in indices]

    def _numeric_values(self, indices: list[int], column: str) -> list[float]:
        values: list[float] = []
        for index in indices:
            value = self.data[index][column]
            if value is None:
                continue
            values.append(float(value))
        return values

    @staticmethod
    def _missing_count(data: Data, column: str) -> int:
        return sum(1 for row in data if row[column] is None)

    def _missing_count_indices(self, indices: list[int], column: str) -> int:
        return sum(1 for index in indices if self.data[index][column] is None)

    def _drop_data_rows(self, drop_indices: set[int]) -> None:
        kept_indices = [
            index for index in range(len(self.data)) if index not in drop_indices
        ]
        old_to_new = {
            old_index: new_index for new_index, old_index in enumerate(kept_indices)
        }

        self.data = [self.data[index] for index in kept_indices]
        self.train_indices = [
            old_to_new[index] for index in self.train_indices if index in old_to_new
        ]
        if self.val_indices is not None:
            self.val_indices = [
                old_to_new[index] for index in self.val_indices if index in old_to_new
            ]
        self.test_indices = [
            old_to_new[index] for index in self.test_indices if index in old_to_new
        ]

    @staticmethod
    def _split_counts(
        size: int, val: bool, val_size: float, test_size: float
    ) -> tuple[int, int]:
        test_count = int(round(size * test_size))
        val_count = int(round(size * val_size)) if val else 0

        if test_count == 0:
            test_count = 1
        if val and val_count == 0:
            val_count = 1

        while test_count + val_count >= size:
            if val and val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            elif val and val_count > 0:
                val_count -= 1
            else:
                break

        return val_count, test_count

    def split_columns(
        self,
        numerical: list[str],
        ordinal: dict[str, list[str]],
        nominal: list[str],
        text: list[str],
        target: list[str],
    ) -> None:
        """Step 1: Split columns by their type.
        Next step -> split_sets
        """
        current_step = 1
        self._check_automatic_step_method_possibility(current_step)

        ordinal_columns = list(ordinal.keys())
        groups: list[tuple[str, list[str]]] = [
            ("numerical", numerical),
            ("ordinal", ordinal_columns),
            ("nominal", nominal),
            ("text", text),
            ("target", target),
        ]

        listed_columns: list[str] = []
        for role, columns in groups:
            for column in columns:
                self._is_column_exists(column)
                listed_columns.append(column)

        duplicate_columns = sorted(
            column for column, count in Counter(listed_columns).items() if count > 1
        )
        if duplicate_columns:
            raise MismatchError(
                f"Columns assigned to more than one role: {', '.join(duplicate_columns)}"
            )

        listed_set = set(listed_columns)
        uncovered_columns = [
            column for column in self.columns if column not in listed_set
        ]
        if uncovered_columns:
            raise MismatchError(
                f"Columns not assigned to any role: {', '.join(uncovered_columns)}"
            )

        if not target:
            raise MismatchError("At least one target column is required!")

        for column, order in ordinal.items():
            if not order:
                raise MismatchError(
                    f'Ordinal column "{column}" requires an ordered category list!'
                )

            keys = [self._category_key(value) for value in order]
            if len(keys) != len(set(keys)):
                raise MismatchError(
                    f'Ordinal order for column "{column}" contains duplicate categories!'
                )

            observed = {
                self._category_key(row[column])
                for row in self.data
                if row[column] is not None
            }
            missing_categories = sorted(observed - set(keys))
            if missing_categories:
                raise MismatchError(
                    f'Ordinal order for column "{column}" is missing categories: '
                    f"{', '.join(missing_categories)}"
                )

        self.numerical = list(numerical)
        self.ordinal = ordinal_columns
        self.nominal = list(nominal)
        self.text = list(text)
        self.target = list(target)
        self.ordinal_order = {column: list(order) for column, order in ordinal.items()}

        self.step = current_step

    def split_sets(
        self,
        val: bool = False,
        val_size: float = 0.2,
        test_size: float = 0.2,
        stratify: bool = False,
    ) -> None:
        """Step 2: Split dataset on train, validation, and test.
        Next step -> show_missing
        """
        current_step = 2
        self._check_automatic_step_method_possibility(current_step)

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1!")
        if val and not 0 < val_size < 1:
            raise ValueError("val_size must be between 0 and 1!")
        if val and val_size + test_size >= 1:
            raise ValueError("val_size + test_size must be less than 1!")

        duplicate_indexes = self.duplicates("exact")
        if duplicate_indexes:
            raise MismatchError(
                "Dataset contains duplicate rows at indexes "
                f"{duplicate_indexes}. Call reset_phase(), drop them, and retry!"
            )

        missing_targets: dict[str, list[int]] = {}
        for column in self.target:
            indexes = self.empty(column)
            if indexes:
                missing_targets[column] = indexes

        if missing_targets:
            details = "; ".join(
                f"{column}: {indexes}" for column, indexes in missing_targets.items()
            )
            raise MismatchError(
                "Target columns contain missing values "
                f"({details}). Call reset_phase(), drop them, and retry!"
            )

        size = len(self.data)
        if size < 2:
            raise MismatchError("Dataset needs at least two rows to split!")
        if val and size < 3:
            raise MismatchError(
                "Dataset needs at least three rows to create a validation set!"
            )

        rng = random.Random(self.seed)
        train_indexes: list[int] = []
        val_indexes: list[int] = []
        test_indexes: list[int] = []

        if stratify:
            if len(self.target) != 1:
                raise MismatchError(
                    "Stratified splitting requires exactly one target column!"
                )

            target_column = self.target[0]
            groups: dict[str, list[int]] = {}
            for index, row in enumerate(self.data):
                key = self._category_key(row[target_column])
                groups.setdefault(key, []).append(index)

            for key, indexes in groups.items():
                if len(indexes) < 2:
                    raise MismatchError(
                        f'Target category "{key}" has fewer than two rows, '
                        "so stratification is not possible!"
                    )

                rng.shuffle(indexes)
                group_val_count, group_test_count = self._split_counts(
                    len(indexes), val, val_size, test_size
                )
                test_indexes.extend(indexes[:group_test_count])
                val_start = group_test_count
                val_end = val_start + group_val_count
                val_indexes.extend(indexes[val_start:val_end])
                train_indexes.extend(indexes[val_end:])

        else:
            indexes = list(range(size))
            rng.shuffle(indexes)
            val_count, test_count = self._split_counts(size, val, val_size, test_size)
            test_indexes = indexes[:test_count]
            val_indexes = indexes[test_count : test_count + val_count]
            train_indexes = indexes[test_count + val_count :]

        if not train_indexes:
            raise MismatchError("Split produced an empty train set!")
        if not test_indexes:
            raise MismatchError("Split produced an empty test set!")
        if val and not val_indexes:
            raise MismatchError("Split produced an empty validation set!")

        self.train_indices = train_indexes
        self.val_indices = val_indexes if val else None
        self.test_indices = test_indexes

        self.step = current_step

    def show_missing(self) -> None:
        """Step 3: Check how many missing cells per column are in the dataset.
        Next step -> plot_distribution
        """
        current_step = 3
        self._check_automatic_step_method_possibility(current_step)

        width = max((len(column) for column in self.columns), default=6)
        if self.val_indices is not None:
            header = (
                f"{'column':<{width}}  {'role':<10} {'train':>8} {'val':>8} {'test':>8}"
            )
        else:
            header = f"{'column':<{width}}  {'role':<10} {'train':>8} {'test':>8}"

        print(header)
        print("-" * len(header))

        for column in self.columns:
            role = self._column_role(column) or "encoded"
            train_missing = self._missing_count_indices(self.train_indices, column)
            test_missing = self._missing_count_indices(self.test_indices, column)

            if self.val_indices is not None:
                val_missing = self._missing_count_indices(self.val_indices, column)
                print(
                    f"{column:<{width}}  {role:<10} {train_missing:>8} "
                    f"{val_missing:>8} {test_missing:>8}"
                )
            else:
                print(
                    f"{column:<{width}}  {role:<10} "
                    f"{train_missing:>8} {test_missing:>8}"
                )

        self.step = current_step

    def _plot_column(self, axis: Axes, data: Data, column: str, bins: int = 10) -> None:
        role = self._column_role(column)
        values = [row[column] for row in data]
        non_missing_values = [value for value in values if value is not None]
        is_numeric = role == "numerical" or (
            role == "target"
            and all(isinstance(value, (int, float)) for value in non_missing_values)
        )

        if is_numeric:
            numbers = [float(value) for value in non_missing_values]
            if numbers:
                axis.hist(numbers, bins=bins, edgecolor="white")
            else:
                axis.text(
                    0.5,
                    0.5,
                    "no numeric values",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
        else:
            labels = [
                "(missing)" if value is None else self._category_key(value)
                for value in values
            ]
            counts = Counter(labels)

            if len(counts) > 15:
                top = counts.most_common(14)
                other_count = sum(counts.values()) - sum(count for _, count in top)
                labels_list = [label for label, _ in top] + ["(other)"]
                counts_list = [count for _, count in top] + [other_count]
            else:
                labels_list = list(counts.keys())
                counts_list = list(counts.values())

            axis.bar(labels_list, counts_list)
            axis.tick_params(axis="x", rotation=45)

        axis.set_title(column)
        axis.set_xlabel(column)

    def plot_distribution(self) -> None:
        """Step 4: Show distribution for all columns in one figure, where numerical columns get hist plot and categorical columns get bar plot.
        Next step -> drop_outliers
        """
        current_step = 4
        self._check_automatic_step_method_possibility(current_step)

        if not self.columns:
            raise MismatchError("Dataset has no columns to plot!")

        plot_columns = list(self.columns)
        columns_per_row = min(4, len(plot_columns))
        rows = math.ceil(len(plot_columns) / columns_per_row)
        figure, axes = plt.subplots(
            rows,
            columns_per_row,
            figsize=(4 * columns_per_row, 3 * rows),
            squeeze=False,
        )

        train_rows = self._rows(self.train_indices)
        for axis, column in zip(axes.flat, plot_columns):
            self._plot_column(axis, train_rows, column)

        for axis in axes.flat[len(plot_columns) :]:
            axis.axis("off")

        figure.suptitle("Train set distributions")
        figure.tight_layout()
        plt.show()

        self.step = current_step

    def drop_outliers(self, sigma: float = 3) -> int:
        """Step 5: Drop train rows with numbers outside mean +/- sigma * std range.
        Removed rows are deleted from the data and split indexes are remapped.
        Next step -> fill_missing
        """
        current_step = 5
        self._check_automatic_step_method_possibility(current_step)

        if sigma <= 0:
            raise ValueError("sigma must be greater than zero!")

        bounds: dict[str, tuple[float, float]] = {}
        for column in self.numerical:
            values = self._numeric_values(self.train_indices, column)
            if not values:
                continue

            column_mean = mean(values)
            column_std = pstdev(values)
            bounds[column] = (
                column_mean - sigma * column_std,
                column_mean + sigma * column_std,
            )

        drop_indices: set[int] = set()
        for index in self.train_indices:
            row = self.data[index]
            for column, (lower, upper) in bounds.items():
                value = row[column]
                if value is None:
                    continue

                number = float(value)
                if number < lower or number > upper:
                    drop_indices.add(index)
                    break

        dropped = len(drop_indices)
        if drop_indices:
            self._drop_data_rows(drop_indices)

        self.step = current_step
        return dropped

    def fill_missing(
        self, mode: Literal["mean", "median", "mode"], columns: list[str]
    ) -> None:
        """Step 6: Mean/median/mode are calculated on train rows and used to fill all rows.
        Next step -> scale_columns
        """
        current_step = 6
        self._check_automatic_step_method_possibility(current_step)

        if mode not in ("mean", "median", "mode"):
            raise ValueError(
                f'Invalid mode "{mode}", expected "mean", "median", or "mode"!'
            )
        if not columns:
            raise ValueError("At least one column is required!")

        fill_values: dict[str, Cell] = {}
        for column in columns:
            self._is_column_exists(column)

            if column in self.numerical:
                if mode == "mode":
                    raise ValueError(
                        f'Numerical column "{column}" requires mean or median filling!'
                    )

                values = self._numeric_values(self.train_indices, column)
                if not values:
                    raise MismatchError(
                        f'Column "{column}" has no numeric train values to fit!'
                    )
                fill_value: Cell = mean(values) if mode == "mean" else median(values)
            else:
                categorical_values = [
                    self.data[index][column]
                    for index in self.train_indices
                    if self.data[index][column] is not None
                ]
                if not categorical_values:
                    raise MismatchError(
                        f'Column "{column}" has no train values to fit!'
                    )
                fill_value = Counter(categorical_values).most_common(1)[0][0]

            fill_values[column] = fill_value

        for row in self.data:
            for column, fill_value in fill_values.items():
                if row[column] is None:
                    row[column] = fill_value

        self.fill_values.update(fill_values)
        self.step = current_step

    def scale_columns(self) -> None:
        """Step 7: Use train mean and std to scale numerical columns in all rows.
        Next step -> encode_columns
        """
        current_step = 7
        self._check_automatic_step_method_possibility(current_step)

        scale_params: dict[str, tuple[float, float]] = {}
        for column in self.numerical:
            values = self._numeric_values(self.train_indices, column)
            if not values:
                continue

            column_mean = mean(values)
            column_std = pstdev(values)
            if column_std == 0:
                column_std = 1.0
            scale_params[column] = (column_mean, column_std)

        for row in self.data:
            for column, (column_mean, column_std) in scale_params.items():
                value = row[column]
                if value is None:
                    continue
                row[column] = (float(value) - column_mean) / column_std

        self.scale_params.update(scale_params)
        self.step = current_step

    def encode_columns(self, min_frequency: int = 5) -> None:
        """Step 8: Label encode ordinal columns, one-hot encode nominal columns, and label encode categorical target columns.
        Encoders fit on train rows only.
        Next step -> show_summary
        """
        current_step = 8
        self._check_automatic_step_method_possibility(current_step)

        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1!")

        ordinal_mappings: dict[str, dict[str, int]] = {}
        for column in self.ordinal:
            order = self.ordinal_order.get(column)
            if order is None:
                raise MismatchError(
                    f'Ordinal column "{column}" has no stored category order!'
                )

            mapping = {
                self._category_key(value): index for index, value in enumerate(order)
            }

            for row in self.data:
                value = row[column]
                if value is None:
                    raise MismatchError(
                        f'Ordinal column "{column}" still contains missing values!'
                    )

                key = self._category_key(value)
                if key not in mapping:
                    raise MismatchError(
                        f'Ordinal column "{column}" has unseen category "{key}"!'
                    )

            ordinal_mappings[column] = mapping

        target_plans: dict[str, dict[str, int] | None] = {}
        for column in self.target:
            train_values = [
                self.data[index][column]
                for index in self.train_indices
                if self.data[index][column] is not None
            ]
            if not train_values:
                raise MismatchError(
                    f'Target column "{column}" has no train values to fit!'
                )

            if any(row[column] is None for row in self.data):
                raise MismatchError(
                    f'Target column "{column}" still contains missing values!'
                )

            if all(isinstance(value, (int, float)) for value in train_values):
                target_plans[column] = None
                continue

            mapping = {
                key: index
                for index, key in enumerate(
                    sorted({self._category_key(value) for value in train_values})
                )
            }

            for row in self.data:
                key = self._category_key(row[column])
                if key not in mapping:
                    raise MismatchError(
                        f'Target column "{column}" has unseen category "{key}"!'
                    )

            target_plans[column] = mapping

        nominal_plans: list[tuple[str, list[str], str, list[str]]] = []
        planned_columns = set(self.columns)
        for column in self.nominal:
            train_values = [
                self._category_key(self.data[index][column])
                for index in self.train_indices
            ]
            counts = Counter(train_values)
            frequent_categories = sorted(
                value for value, count in counts.items() if count >= min_frequency
            )
            other_column = f"{column}=__other__"
            new_columns = [f"{column}={category}" for category in frequent_categories]
            new_columns.append(other_column)

            collisions = [
                new_column
                for new_column in new_columns
                if new_column in planned_columns and new_column != column
            ]
            if collisions:
                raise MismatchError(
                    f"One-hot encoding would create existing columns: "
                    f"{', '.join(collisions)}"
                )

            planned_columns.discard(column)
            planned_columns.update(new_columns)
            nominal_plans.append(
                (column, frequent_categories, other_column, new_columns)
            )

        for column, mapping in ordinal_mappings.items():
            for row in self.data:
                row[column] = mapping[self._category_key(row[column])]

        for column, frequent_categories, other_column, new_columns in nominal_plans:
            for row in self.data:
                value = row[column]
                key = "__missing__" if value is None else self._category_key(value)
                for category in frequent_categories:
                    row[f"{column}={category}"] = int(key == category)
                row[other_column] = int(key not in frequent_categories)
                del row[column]

            self.columns.remove(column)
            self.columns.extend(new_columns)

        for column, mapping in target_plans.items():
            if mapping is None:
                continue

            for row in self.data:
                row[column] = mapping[self._category_key(row[column])]

            self.target_mappings[column] = mapping

        self.step = current_step

    def show_summary(self) -> None:
        """Final step: Inspect the full transformed dataset's types, missing counts, and first five rows."""
        current_step = 9
        self._check_automatic_step_method_possibility(current_step)

        width = max((len(column) for column in self.columns), default=6)
        split_sizes = f"train={len(self.train_indices)}, test={len(self.test_indices)}"
        if self.val_indices is not None:
            split_sizes = (
                f"train={len(self.train_indices)}, "
                f"val={len(self.val_indices)}, "
                f"test={len(self.test_indices)}"
            )

        print("Dataset summary")
        print(f"Rows: {len(self.data)} ({split_sizes})")
        print(f"{'column':<{width}}  {'role':<10} {'type':<12} {'missing':>8}")
        print("-" * (width + 35))

        for column in self.columns:
            role = self._column_role(column) or "encoded"
            types = sorted({type(row[column]).__name__ for row in self.data})
            type_name = "/".join(types) if types else "-"
            missing = self._missing_count(self.data, column)
            print(f"{column:<{width}}  {role:<10} {type_name:<12} {missing:>8}")

        print()
        self._print_head(self.data, self.columns, 5)

        self.step = current_step

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def set_none_values(self, none_values: Iterable[str]) -> None:
        self._check_write_method_possibility()

        normalized_none_values = self._normalize_none_values(none_values)
        self.none_values = normalized_none_values

        for row in self.data:
            for column in self.columns:
                value = row[column]
                if (
                    isinstance(value, str)
                    and value.strip().lower() in normalized_none_values
                ):
                    row[column] = None

    def switch_phase(self) -> None:
        if self.phase != "manual":
            raise PhaseMismatchError(
                "Transition to automatic phase is only possible from manual phase!"
            )

        self._manual_snapshot = (self._copy_data(self.data), list(self.columns))
        self.phase = "automatic"
        self.step = 0

    def reset_phase(self) -> None:
        if self.phase != "automatic":
            raise PhaseMismatchError(
                "Reset to manual phase is only possible from automatic phase!"
            )
        if self._manual_snapshot is None:
            raise PhaseMismatchError("No manual state snapshot is available!")

        data, columns = self._manual_snapshot
        self.data = self._copy_data(data)
        self.columns = list(columns)
        self.phase = "manual"
        self.step = 0
        self.numerical = []
        self.ordinal = []
        self.nominal = []
        self.text = []
        self.target = []
        self.ordinal_order = {}
        self.train_indices = []
        self.val_indices = None
        self.test_indices = []
        self.fill_values = {}
        self.scale_params = {}
        self.target_mappings = {}
        self._manual_snapshot = None
