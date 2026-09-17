"""
WorkingNumericTarget and quality functions for numeric subgroup discovery.

This module mirrors the structure of pysubgroup's ``binary_target`` module, but
for a *numeric* (real-valued) target attribute.  It provides:

    * ``WorkingNumericTarget``       -- the target definition and its statistics.
    * ``SimpleNumericTargetQF``       -- shared machinery (constant/subgroup
                                         statistics, GP-Growth hooks) for the
                                         mean-based quality functions.
    * ``StandardQFNumeric``           -- the general, parameterised quality
                                         function ``n_sg**a * (mean_sg - mean_ds)``.
    * ``MeanQF`` (a=0), ``SimpleBinomialQFNumeric`` (a=0.5),
      ``WRAccQFNumeric`` (a=1)        -- the classic order-equivalent members of
                                         the standard family.
    * ``MedianQFNumeric``             -- a robust, median-based alternative.

The ``StandardQFNumeric`` family is the numeric analogue of pysubgroup's
``StandardQF`` family: the exponent ``a`` trades off subgroup *size* against the
*deviation* of the subgroup mean from the overall mean.

@author: adapted from the pysubgroup BinaryTarget module
"""
import numbers
from collections import namedtuple
from functools import total_ordering

import numpy as np

from pysubgroup.measures import (
    AbstractInterestingnessMeasure,
    BoundedInterestingnessMeasure,
)

from .subgroup_description import get_cover_array_and_size
from .utils import BaseTarget


def _resolve_cover(subgroup, data, data_len):
    """Return a ``(cover, size)`` pair suitable for indexing the target values.

    The fast path returns exactly what pysubgroup's ``get_cover_array_and_size``
    produces (a boolean array, a slice, or an array-like representation object).
    Only when the helper hands back an *opaque* subgroup object that numpy cannot
    use as an index -- e.g. an interned selector still carrying a stale
    ``representation`` from a previous algorithm run and used outside that
    algorithm's representation context -- do we recompute a plain boolean mask
    from the raw data, so computing statistics interactively never crashes.
    """
    cover_arr, size = get_cover_array_and_size(subgroup, data_len, data)
    if isinstance(cover_arr, (np.ndarray, slice)) or hasattr(
        cover_arr, "__array_interface__"
    ):
        return cover_arr, size
    mask = subgroup.covers(data)
    return mask, int(np.count_nonzero(mask))


def _safe_ratio(numerator, denominator):
    """Return ``numerator / denominator``, or ``nan`` if the denominator is 0/nan.

    Used for the ``*_lift`` statistics, where a zero or undefined overall
    centroid would otherwise raise or return an infinity.
    """
    if denominator == 0 or np.isnan(denominator):
        return float("nan")
    return numerator / denominator


@total_ordering
class WorkingNumericTarget(BaseTarget):
    """Numeric target for subgroup discovery with a real-valued target attribute.

    Stores the target attribute and computes distribution statistics of the
    target inside a subgroup versus the whole dataset: size, mean, standard
    deviation, median, extrema, and the mean/median lifts (subgroup value
    divided by the dataset value).
    """

    statistic_types = (
        "size_sg",
        "size_dataset",
        "mean_sg",
        "mean_dataset",
        "std_sg",
        "std_dataset",
        "median_sg",
        "median_dataset",
        "max_sg",
        "max_dataset",
        "min_sg",
        "min_dataset",
        "mean_lift",
        "median_lift",
    )

    def __init__(self, target_variable):
        """Initialize a WorkingNumericTarget instance.

        Parameters:
            target_variable (str): The name of the numeric target attribute.
        """
        self.target_variable = target_variable

    def __repr__(self):
        """String representation of the WorkingNumericTarget."""
        return "T: " + str(self.target_variable)

    def __eq__(self, other):
        """Check equality based on the instance dictionary."""
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        """Define less-than comparison for sorting purposes."""
        return str(self) < str(other)

    def get_attributes(self):
        """Get the attribute names used in the target.

        Returns:
            tuple: A tuple containing the target attribute name.
        """
        return (self.target_variable,)

    def get_base_statistics(self, subgroup, data):
        """Compute basic statistics (sizes and means) for the subgroup and dataset.

        Parameters:
            subgroup: The subgroup for which to compute statistics.
            data (pandas.DataFrame): The dataset.

        Returns:
            tuple: (instances_dataset, mean_dataset,
                    instances_subgroup, mean_subgroup).
        """
        cover_arr, size_sg = _resolve_cover(subgroup, data, len(data))
        all_target_values = data[self.target_variable].to_numpy()
        sg_target_values = all_target_values[cover_arr]
        instances_dataset = len(data)
        instances_subgroup = size_sg
        mean_dataset = np.mean(all_target_values)
        mean_sg = np.mean(sg_target_values) if size_sg > 0 else float("nan")
        return (instances_dataset, mean_dataset, instances_subgroup, mean_sg)

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        """Calculate the full set of ``statistic_types`` for the subgroup.

        Parameters:
            subgroup: The subgroup for which to calculate statistics.
            data (pandas.DataFrame): The dataset.
            cached_statistics (dict, optional): Previously computed statistics.

        Returns:
            dict: A dictionary containing every entry of ``statistic_types``.
        """
        if self.all_statistics_present(cached_statistics):
            return cached_statistics

        cover_arr, size_sg = _resolve_cover(subgroup, data, len(data))
        all_target_values = data[self.target_variable].to_numpy()
        sg_target_values = all_target_values[cover_arr]

        statistics = {}
        statistics["size_sg"] = size_sg
        statistics["size_dataset"] = len(data)
        statistics["mean_dataset"] = np.mean(all_target_values)
        statistics["std_dataset"] = np.std(all_target_values)
        statistics["median_dataset"] = np.median(all_target_values)
        statistics["max_dataset"] = np.max(all_target_values)
        statistics["min_dataset"] = np.min(all_target_values)

        if size_sg > 0:
            statistics["mean_sg"] = np.mean(sg_target_values)
            statistics["std_sg"] = np.std(sg_target_values)
            statistics["median_sg"] = np.median(sg_target_values)
            statistics["max_sg"] = np.max(sg_target_values)
            statistics["min_sg"] = np.min(sg_target_values)
        else:  # empty subgroup: report NaN rather than raising on empty arrays
            statistics["mean_sg"] = float("nan")
            statistics["std_sg"] = float("nan")
            statistics["median_sg"] = float("nan")
            statistics["max_sg"] = float("nan")
            statistics["min_sg"] = float("nan")

        statistics["mean_lift"] = _safe_ratio(
            statistics["mean_sg"], statistics["mean_dataset"]
        )
        statistics["median_lift"] = _safe_ratio(
            statistics["median_sg"], statistics["median_dataset"]
        )
        return statistics


class WorkingSimpleNumericTargetQF(AbstractInterestingnessMeasure):
    # pylint: disable=abstract-method
    """Shared machinery for mean-based quality functions on a numeric target.

    Precomputes the dataset size, mean and standard deviation, holds the raw
    target values, and computes per-subgroup statistics (size, mean, and an
    optimistic-estimate value).  It also provides the GP-Growth hooks so that
    every subclass works with the ``GpGrowth`` algorithm out of the box.

    Concrete quality functions subclass this and implement ``evaluate``
    (and, optionally, ``optimistic_estimate``).
    """

    # per-subgroup statistics passed around by the algorithms
    tpl = namedtuple("NumericTargetQF_parameters", ("size_sg", "mean", "estimate"))
    # dataset-wide statistics computed once
    dataset_tpl = namedtuple("NumericTargetQF_global", ("size_sg", "mean", "std"))

    def __init__(self):
        """Initialize the shared statistics containers."""
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False
        self.required_stat_attrs = ("size_sg", "mean")
        # subclasses that support a search for *low* values flip this to True
        self.invert = False

    def calculate_constant_statistics(self, data, target):
        """Calculate the statistics that are constant for the whole dataset.

        Parameters:
            data (pandas.DataFrame): The dataset.
            target (WorkingNumericTarget): The target definition.

        Raises:
            AssertionError: If the target is not a WorkingNumericTarget.
        """
        assert isinstance(target, WorkingNumericTarget)
        self.all_target_values = data[target.target_variable].to_numpy()
        self.dataset_statistics = WorkingSimpleNumericTargetQF.dataset_tpl(
            len(data),
            np.mean(self.all_target_values),
            np.std(self.all_target_values),
        )
        self.has_constant_statistics = True

    def calculate_statistics(
        self, subgroup, target, data, statistics=None
    ):  # pylint: disable=unused-argument
        """Calculate statistics specific to the subgroup.

        Parameters:
            subgroup: The subgroup for which to calculate statistics.
            target (WorkingNumericTarget): The target definition.
            data (pandas.DataFrame): The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            namedtuple: (size_sg, mean, estimate) for the subgroup. ``estimate``
                        is a valid optimistic estimate for any ``a <= 1``.
        """
        cover_arr, size_sg = _resolve_cover(
            subgroup, data, len(self.all_target_values)
        )
        if size_sg > 0:
            sg_values = self.all_target_values[cover_arr]
            sg_mean = np.mean(sg_values)
            estimate = self._summation_estimate(sg_values)
        else:
            sg_mean = float("nan")
            estimate = float("-inf")
        return WorkingSimpleNumericTargetQF.tpl(size_sg, sg_mean, estimate)

    def _summation_estimate(self, sg_values):
        """Sum of the positive deviations of ``sg_values`` from the dataset mean.

        This quantity, ``sum_{x in sg, dev(x) > 0} dev(x)``, is a valid
        optimistic estimate for ``StandardQFNumeric`` whenever ``a <= 1``:
        for any refinement ``S'`` of the subgroup,
        ``|S'|**a * (mean(S') - mu) <= |S'|**(a-1) * sum_{S'} dev(x)``
        ``<= sum_{x in sg, dev(x) > 0} dev(x)`` because ``|S'|**(a-1) <= 1`` for
        ``a <= 1`` and dropping negative-deviation instances only increases the
        sum.  For ``invert=True`` the deviation is measured in the opposite
        direction (searching for *low*-mean subgroups).
        """
        data_mean = self.dataset_statistics.mean
        if self.invert:
            deviations = data_mean - sg_values
        else:
            deviations = sg_values - data_mean
        return float(np.sum(deviations[deviations > 0]))

    # <<< GP-Growth >>>
    # Each row contributes [count, value, positive_deviation]. The positive
    # deviation ``max(0, signed_dev)`` is measured against the *constant* global
    # mean, so it is additively decomposable and its group sum equals the
    # optimistic estimate of ``_summation_estimate`` -- this lets GP-Growth prune
    # from the aggregated node statistics alone.
    def gp_get_stats(self, row_index):
        """Get statistics for a single row (used in GP-Growth algorithms).

        Returns:
            numpy.ndarray: Array ``[1, value, positive_deviation]`` for the row.
        """
        value = self.all_target_values[row_index]
        data_mean = self.dataset_statistics.mean
        signed_dev = (data_mean - value) if self.invert else (value - data_mean)
        return np.array([1.0, value, max(0.0, signed_dev)], dtype=float)

    def gp_get_null_vector(self):
        """Get a null vector for initialization in GP-Growth algorithms.

        Returns:
            numpy.ndarray: Zeros of size 3 (count, value-sum, positive-dev-sum).
        """
        return np.zeros(3)

    def gp_merge(self, left, right):
        """Merge two statistics vectors by summing all components."""
        left += right

    def gp_get_params(self, _cover_arr, v):
        """Extract (size_sg, mean, estimate) from a GP-Growth statistics vector.

        The vector holds ``[count, sum_of_values, sum_of_positive_deviations]``;
        the mean is ``sum / count`` and the optimistic estimate is the third
        component (a valid bound for ``a <= 1``).
        """
        size = v[0]
        mean = v[1] / v[0] if v[0] > 0 else float("nan")
        return WorkingSimpleNumericTargetQF.tpl(size, mean, v[2])

    def gp_to_str(self, stats):
        """Convert a statistics vector to a string representation."""
        return " ".join(map(str, stats))

    def gp_size_sg(self, stats):
        """Get the subgroup size (the count) from a statistics vector."""
        return stats[0]

    @property
    def gp_requires_cover_arr(self):
        """Indicate whether GP-Growth requires a cover array.

        Returns:
            bool: False, since the cover array is not required.
        """
        return False


class WorkingStandardQFNumeric(WorkingSimpleNumericTargetQF, BoundedInterestingnessMeasure):
    """StandardQF for numeric targets: size traded off against mean deviation.

    Computes ``n_sg**a * (mean_sg - mean_dataset)``.  Like the binary
    ``StandardQF``, different values of ``a`` are order-equivalent to several
    popular quality measures (see ``MeanQF``, ``SimpleBinomialQFNumeric`` and
    ``WRAccQFNumeric``).  With ``invert=True`` the sign of the deviation is
    flipped, so the search rewards subgroups whose mean is *below* the overall
    mean.
    """

    @staticmethod
    def standard_qf_numeric(
        a, mean_dataset, instances_subgroup, mean_subgroup
    ):
        """Compute the standard numeric quality function.

        Parameters:
            a (float): Exponent trading off subgroup size against mean deviation.
            mean_dataset (float): Mean of the target over the whole dataset.
            instances_subgroup (int): Number of instances in the subgroup.
            mean_subgroup (float): Mean of the target within the subgroup.

        Returns:
            float: The quality value, or ``nan`` for an empty subgroup.
        """
        if not hasattr(instances_subgroup, "__array_interface__") and (
            instances_subgroup == 0
        ):
            return float("nan")
        return instances_subgroup**a * (mean_subgroup - mean_dataset)

    def __init__(self, a, invert=False):
        """Initialize the StandardQFNumeric.

        Parameters:
            a (float): Exponent trading off subgroup size against mean deviation.
            invert (bool, optional): If True, reward subgroups with a *low* mean.

        Raises:
            ValueError: If ``a`` is not a number.
        """
        if not isinstance(a, numbers.Number):
            raise ValueError(f"a is expected to be a number, got {a!r}")
        super().__init__()
        self.a = a
        self.invert = invert

    def evaluate(self, subgroup, target, data, statistics=None):
        """Evaluate the quality of the subgroup using the standard numeric QF.

        Parameters:
            subgroup: The subgroup to evaluate.
            target (WorkingNumericTarget): The target definition.
            data (pandas.DataFrame): The dataset.
            statistics (any, optional): Cached statistics, if available.

        Returns:
            float: The computed quality value.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        result = WorkingStandardQFNumeric.standard_qf_numeric(
            self.a, dataset.mean, statistics.size_sg, statistics.mean
        )
        return -result if self.invert else result

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        """Compute an optimistic estimate of the quality function.

        The estimate (sum of positive deviations within the subgroup) is a
        valid upper bound on the quality of every refinement whenever
        ``a <= 1``.  For ``a > 1`` no cheap valid bound is available, so
        ``inf`` is returned (correct, but disables pruning).

        Parameters:
            subgroup: The subgroup for which to compute the estimate.
            target (WorkingNumericTarget): The target definition.
            data (pandas.DataFrame): The dataset.
            statistics (any, optional): Cached statistics, if available.

        Returns:
            float: The optimistic estimate of the quality value.
        """
        if self.a > 1:
            return float("inf")
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate


class WorkingMeanQF(WorkingStandardQFNumeric):
    """Mean Quality Function.

    ``MeanQF`` is a ``StandardQFNumeric`` with ``a=0``.  It scores a subgroup
    purely by the deviation of its mean from the dataset mean, ignoring the
    subgroup size entirely (the numeric analogue of ``LiftQF``).
    """

    def __init__(self, invert=False):
        """Initialize the MeanQF."""
        super().__init__(0.0, invert=invert)


class WorkingSimpleBinomialQFNumeric(WorkingStandardQFNumeric):
    """Simple Binomial Quality Function for numeric targets.

    ``SimpleBinomialQFNumeric`` is a ``StandardQFNumeric`` with ``a=0.5``.  It
    balances subgroup size against mean deviation and is order-equivalent to a
    z-score of the subgroup mean against the dataset mean.
    """

    def __init__(self, invert=False):
        """Initialize the SimpleBinomialQFNumeric."""
        super().__init__(0.5, invert=invert)


class WorkingWRAccQFNumeric(WorkingStandardQFNumeric):
    """Weighted Relative Accuracy (impact) Quality Function for numeric targets.

    ``WRAccQFNumeric`` is a ``StandardQFNumeric`` with ``a=1``.  It is
    order-equivalent to the difference between the observed and expected sum of
    the target over the subgroup, i.e. the subgroup's total *impact*.
    """

    def __init__(self, invert=False):
        """Initialize the WRAccQFNumeric."""
        super().__init__(1.0, invert=invert)


class WorkingMedianQFNumeric(AbstractInterestingnessMeasure):
    # pylint: disable=abstract-method
    """Robust, median-based quality function for numeric targets.

    Computes ``n_sg**a * (median_sg - median_dataset)``.  Because the median is
    not decomposable, this measure supports neither GP-Growth nor an optimistic
    estimate, but it is far less sensitive to outliers than the mean-based
    ``StandardQFNumeric`` family.
    """

    tpl = namedtuple("MedianQFNumeric_parameters", ("size_sg", "median"))
    dataset_tpl = namedtuple("MedianQFNumeric_global", ("size_sg", "median"))

    def __init__(self, a=1.0, invert=False):
        """Initialize the MedianQFNumeric.

        Parameters:
            a (float, optional): Exponent trading off size against deviation.
            invert (bool, optional): If True, reward subgroups with a low median.
        """
        if not isinstance(a, numbers.Number):
            raise ValueError(f"a is expected to be a number, got {a!r}")
        self.a = a
        self.invert = invert
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False
        self.required_stat_attrs = ("size_sg", "median")

    def calculate_constant_statistics(self, data, target):
        """Calculate the dataset-wide median statistics."""
        assert isinstance(target, WorkingNumericTarget)
        self.all_target_values = data[target.target_variable].to_numpy()
        self.dataset_statistics = WorkingMedianQFNumeric.dataset_tpl(
            len(data), np.median(self.all_target_values)
        )
        self.has_constant_statistics = True

    def calculate_statistics(
        self, subgroup, target, data, statistics=None
    ):  # pylint: disable=unused-argument
        """Calculate the subgroup size and median."""
        cover_arr, size_sg = _resolve_cover(
            subgroup, data, len(self.all_target_values)
        )
        if size_sg > 0:
            median_sg = np.median(self.all_target_values[cover_arr])
        else:
            median_sg = float("nan")
        return WorkingMedianQFNumeric.tpl(size_sg, median_sg)

    def evaluate(self, subgroup, target, data, statistics=None):
        """Evaluate the quality of the subgroup using the median-based QF."""
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        if statistics.size_sg == 0:
            return float("nan")
        deviation = statistics.median - self.dataset_statistics.median
        if self.invert:
            deviation = -deviation
        return statistics.size_sg**self.a * deviation