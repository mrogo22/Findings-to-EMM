"""conformance_target.py

Custom pysubgroup target and quality functions for Exceptional Model Mining
on an ordinal conformance-severity variable.

Contents:
    - ConformanceTarget: numeric/ordinal target reporting distributional
      (AUC / EMD) statistics against dataset and complement.
    - AUCQFNumeric: Mann-Whitney / AUC quality function.
    - EMDQFNumeric: Earth Mover's Distance quality function.

Both quality functions take a `comparison` argument ('dataset' or 'complement')
chosen at runtime, and a size-weighting exponent `a` matching pysubgroup's
StandardQFNumeric semantics.
"""

import numbers
from collections import namedtuple
from functools import total_ordering

import numpy as np
import pysubgroup as ps

@total_ordering
# --------------------------------------------------------------------------- #
# target
# --------------------------------------------------------------------------- #
class ConformanceTarget:
    """Target for an ordinal conformance-severity variable.

    Reports statistics relevant to distributional/ordinal quality functions
    (AUC / EMD) rather than mean-shift: the severity distribution of the
    subgroup and its complement, AUC, EMD, and coverage.
    """

    statistic_types = (
        "size_sg",
        "size_dataset",
        "size_complement",
        "coverage_sg",
        "auc_vs_dataset",
        "auc_vs_complement",
        "emd_vs_dataset",
        "emd_vs_complement",
        "mean_sg",
        "mean_complement",
        "mean_dataset",
        "median_sg",
        "median_complement",
        "median_dataset",
        "dist_sg",
        "dist_complement",
        "dist_dataset",
        "perfect_frac_sg",
        "perfect_frac_dataset",
        "worst_frac_sg",
        "worst_frac_dataset",
    )

    def __init__(self, target_variable):
        """Initialize the ConformanceTarget.

        Parameters:
            target_variable (str): name of the ordinal severity column.
        """
        self.target_variable = target_variable

    def __repr__(self):
        return "T: " + str(self.target_variable)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__  # pragma: no cover

    def __lt__(self, other):
        return str(self) < str(other)  # pragma: no cover

    def get_attributes(self):
        """Return the list of attribute names used by the target."""
        return [self.target_variable]

    def get_base_statistics(self, subgroup, data):
        """Basic (mean-based) statistics, kept for interface compatibility.

        Returns:
            tuple: (instances_dataset, mean_dataset, instances_subgroup, mean_sg)
        """
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(data), data)
        all_target_values = data[self.target_variable]
        sg_target_values = all_target_values[cover_arr]
        instances_dataset = len(data)
        instances_subgroup = size_sg
        mean_sg = np.mean(sg_target_values)
        mean_dataset = np.mean(all_target_values)
        return (instances_dataset, mean_dataset, instances_subgroup, mean_sg)

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        """Calculate conformance-relevant statistics for the subgroup.

        Parameters:
            subgroup: subgroup to evaluate.
            data (pandas.DataFrame): the dataset.
            cached_statistics (dict, optional): previously computed statistics.

        Returns:
            dict: distributional / ordinal statistics.
        """
        if cached_statistics is None or not isinstance(cached_statistics, dict):
            statistics = {}
        elif all(k in cached_statistics for k in ConformanceTarget.statistic_types):
            return cached_statistics
        else:
            statistics = cached_statistics

        cover_arr, _ = ps.get_cover_array_and_size(subgroup, len(data), data)
        all_vals = data[self.target_variable].to_numpy()
        sg_vals = all_vals[cover_arr]
        comp_vals = all_vals[~cover_arr]

        grid = np.unique(all_vals)
        n_sg = len(sg_vals)
        n_comp = len(comp_vals)
        n_all = len(all_vals)
        best = grid.min()
        worst = grid.max()

        def dist(vals):
            counts = {int(k): 0.0 for k in grid}
            if len(vals) == 0:
                return counts
            u, c = np.unique(vals, return_counts=True)
            for k, cnt in zip(u, c):
                counts[int(k)] = cnt / len(vals)
            return counts

        statistics["size_sg"] = n_sg
        statistics["size_dataset"] = n_all
        statistics["size_complement"] = n_comp
        statistics["coverage_sg"] = n_sg / n_all if n_all else 0.0

        statistics["auc_vs_dataset"] = _auc(sg_vals, all_vals) if n_sg else 0.5
        statistics["auc_vs_complement"] = (
            _auc(sg_vals, comp_vals) if n_sg and n_comp else 0.5
        )
        statistics["emd_vs_dataset"] = _emd(sg_vals, all_vals, grid) if n_sg else 0.0
        statistics["emd_vs_complement"] = (
            _emd(sg_vals, comp_vals, grid) if n_sg and n_comp else 0.0
        )

        statistics["mean_sg"] = np.mean(sg_vals) if n_sg else float("nan")
        statistics["mean_complement"] = np.mean(comp_vals) if n_comp else float("nan")
        statistics["mean_dataset"] = np.mean(all_vals)
        statistics["median_sg"] = np.median(sg_vals) if n_sg else float("nan")
        statistics["median_complement"] = (
            np.median(comp_vals) if n_comp else float("nan")
        )
        statistics["median_dataset"] = np.median(all_vals)

        statistics["dist_sg"] = dist(sg_vals)
        statistics["dist_complement"] = dist(comp_vals)
        statistics["dist_dataset"] = dist(all_vals)

        statistics["perfect_frac_sg"] = np.mean(sg_vals == best) if n_sg else float("nan")
        statistics["perfect_frac_dataset"] = np.mean(all_vals == best)
        statistics["worst_frac_sg"] = np.mean(sg_vals == worst) if n_sg else float("nan")
        statistics["worst_frac_dataset"] = np.mean(all_vals == worst)

        return statistics


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ecdf_on_grid(vals, grid):
    """Right-continuous empirical CDF of `vals` evaluated on `grid`.

    Parameters:
        vals (np.ndarray): sample values (need not be sorted).
        grid (np.ndarray): ascending distinct evaluation points.

    Returns:
        np.ndarray: CDF values aligned to `grid`.
    """
    vals = np.sort(vals)
    cum = np.arange(1, len(vals) + 1) / len(vals)
    idx = np.searchsorted(vals, grid, side="right") - 1
    out = np.zeros(len(grid), dtype=float)
    ok = idx >= 0
    out[ok] = cum[idx[ok]]
    return out


def _auc(sg_values, ref_values):
    """AUC = P(sg > ref) + 0.5 * P(sg == ref), via midrank statistic. O(m log m).

    Returns 0.5 when either group is empty (i.e. 'not exceptional').
    """
    n_sg = len(sg_values)
    n_ref = len(ref_values)
    if n_sg == 0 or n_ref == 0:
        return 0.5
    combined = np.concatenate([sg_values, ref_values])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    sorted_c = combined[order]
    i = 0
    n = len(combined)
    while i < n:
        j = i
        while j + 1 < n and sorted_c[j + 1] == sorted_c[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-based midrank
        i = j + 1
    rank_sum_sg = ranks[:n_sg].sum()
    u_sg = rank_sum_sg - n_sg * (n_sg + 1) / 2.0
    return u_sg / (n_sg * n_ref)


def _emd(sg_values, ref_values, grid):
    """Earth Mover's Distance between empirical CDFs over `grid`.

    For a 1-D target this equals the L1 distance between the CDFs, using the
    true numeric gaps between consecutive grid points as ground distance.
    """
    if len(grid) <= 1:
        return 0.0
    gaps = np.diff(grid)
    diff = np.abs(_ecdf_on_grid(sg_values, grid) - _ecdf_on_grid(ref_values, grid))[:-1]
    return float(np.sum(diff * gaps))

def read_median(tpl):
    """Extract the median value from a namedtuple.

    Parameters:
        tpl (namedtuple): A namedtuple containing a 'median' field.

    Returns:
        float: The median value.
    """
    return tpl.median


def read_mean(tpl):
    """Extract the mean value from a namedtuple.

    Parameters:
        tpl (namedtuple): A namedtuple containing a 'mean' field.

    Returns:
        float: The mean value.
    """
    return tpl.mean


def calc_sorted_median(arr):
    """Calculate the median of a sorted array.

    Parameters:
        arr (numpy.ndarray): A sorted array.

    Returns:
        float: The median value.
    """
    half = (len(arr) - 1) // 2
    if len(arr) % 2 == 0:
        return (arr[half] + arr[half + 1]) / 2
    else:
        return arr[half]


# --------------------------------------------------------------------------- #
# quality functions
# --------------------------------------------------------------------------- #
class AUCQFNumeric(ps.BoundedInterestingnessMeasure):
    """Mann-Whitney / AUC quality function for an ordinal target.

    AUC = P(x_sg > x_ref) + 0.5 * P(x_sg = x_ref). Quality is a size-weighted
    deviation of AUC from 0.5. With severity coded 0 (best) .. K (worst),
    AUC < 0.5 means the subgroup is MORE compliant (lower severity) than the
    reference, AUC > 0.5 means LESS compliant.

        phi(sg) = n_sg**a * (2 * |AUC - 0.5|)      [direction 'both']
                = n_sg**a * (2 * (0.5 - AUC))       [direction 'more_compliant']
                = n_sg**a * (2 * (AUC - 0.5))       [direction 'less_compliant']

    Subgroups going against the requested direction score 0. Subgroups outside
    the coverage band [min_sg_coverage, max_sg_coverage] are discarded.

    Parameters:
        a (float): exponent weighting subgroup size.
        comparison (str): 'complement' (natural Mann-Whitney) or 'dataset'.
        direction (str): 'more_compliant', 'less_compliant', or 'both'.
        min_sg_coverage (float): minimum dataset fraction a subgroup must cover.
        max_sg_coverage (float): maximum dataset fraction a subgroup may cover.
        invert (bool): kept for interface parity, unused.
    """

    tpl = namedtuple("AUCQFNumeric_parameters", ("size_sg", "auc", "estimate"))

    def __init__(
        self,
        a,
        comparison="complement",
        direction="both",
        min_sg_coverage=0.0,
        max_sg_coverage=1.0,
        invert=False,
    ):
        if not isinstance(a, numbers.Number):
            raise ValueError(f"a is not a number. Received a={a}")
        if comparison not in ("dataset", "complement"):
            raise ValueError(
                f"comparison was {comparison}, must be 'dataset' or 'complement'"
            )
        if direction not in ("more_compliant", "less_compliant", "both"):
            raise ValueError(
                f"direction was {direction}, must be 'more_compliant', "
                "'less_compliant', or 'both'"
            )
        if not 0.0 <= min_sg_coverage <= max_sg_coverage <= 1.0:
            raise ValueError(
                "require 0 <= min_sg_coverage <= max_sg_coverage <= 1; "
                f"got min={min_sg_coverage}, max={max_sg_coverage}"
            )
        self.a = a
        self.comparison = comparison
        self.direction = direction
        self.min_sg_coverage = min_sg_coverage
        self.max_sg_coverage = max_sg_coverage
        self.invert = invert
        self.required_stat_attrs = ("size_sg", "auc")
        self.all_target_values = None
        self.dataset_size = None
        self.has_constant_statistics = False

    def calculate_constant_statistics(self, data, target):
        """Cache the full target vector and dataset size."""
        self.all_target_values = data[target.target_variable].to_numpy()
        self.dataset_size = len(self.all_target_values)
        self.has_constant_statistics = True

    def _coverage_ok(self, sg_size):
        """True if sg_size falls within the allowed coverage band."""
        cov = sg_size / self.dataset_size
        return self.min_sg_coverage <= cov <= self.max_sg_coverage

    def _directed_deviation(self, auc):
        """Signed, direction-filtered deviation of AUC from 0.5 (>=0, or 0)."""
        if self.direction == "both":
            dev = abs(auc - 0.5)
        elif self.direction == "less_compliant":  # want AUC > 0.5
            dev = max(auc - 0.5, 0.0)
        else:  # more_compliant: want AUC < 0.5
            dev = max(0.5 - auc, 0.0)
        return 2.0 * dev

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        """Compute subgroup size, AUC, and a safe optimistic estimate.

        Out-of-coverage subgroups get auc=0.5 and estimate=-inf.
        """
        cover_arr, sg_size = ps.get_cover_array_and_size(
            subgroup, len(self.all_target_values), data
        )
        if (
            sg_size == 0
            or sg_size == self.dataset_size
            or not self._coverage_ok(sg_size)
        ):
            return self.tpl(sg_size, 0.5, float("-inf"))
        sg_values = self.all_target_values[cover_arr]
        if self.comparison == "complement":
            ref_values = self.all_target_values[~cover_arr]
        else:
            ref_values = self.all_target_values
        auc = _auc(sg_values, ref_values)
        # Safe (admissible, non-pruning) optimistic estimate: max deviation = 0.5
        estimate = sg_size**self.a
        return self.tpl(sg_size, auc, estimate)

    def evaluate(self, subgroup, target, data, statistics=None):
        """Size-weighted, direction-filtered AUC deviation; 0 if filtered out."""
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        if not self._coverage_ok(statistics.size_sg):
            return 0.0
        return statistics.size_sg**self.a * self._directed_deviation(statistics.auc)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        """Return the cached optimistic estimate (loose but admissible)."""
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate

class AUCQFFeatNumeric(ps.BoundedInterestingnessMeasure):
    """AUC quality function weighted by a feature sum instead of subgroup size.

    Identical to AUCQFNumeric except the size-weighting term n_sg**a is replaced
    by (sum of `weight_feature` over the subgroup)**a:

        phi(sg) = W_sg**a * (2 * |AUC - 0.5|)          [direction 'both']
                = W_sg**a * (2 * (0.5 - AUC))           [direction 'more_compliant']
                = W_sg**a * (2 * (AUC - 0.5))           [direction 'less_compliant']

    where W_sg = sum over the subgroup of the values in `weight_feature`.

    Coverage constraints still use the count-based coverage n_sg / n_dataset.

    Parameters:
        a (float): exponent weighting the subgroup feature-sum.
        weight_feature (str): column name whose values are summed for weighting.
        comparison (str): 'complement' (natural Mann-Whitney) or 'dataset'.
        direction (str): 'more_compliant', 'less_compliant', or 'both'.
        min_sg_coverage (float): minimum dataset fraction a subgroup must cover.
        max_sg_coverage (float): maximum dataset fraction a subgroup may cover.
        invert (bool): kept for interface parity, unused.
    """

    tpl = namedtuple(
        "AUCQFFeatNumeric_parameters", ("size_sg", "weight_sg", "auc", "estimate")
    )

    def __init__(
        self,
        a,
        weight_feature,
        comparison="complement",
        direction="both",
        min_sg_coverage=0.0,
        max_sg_coverage=1.0,
        invert=False,
    ):
        if not isinstance(a, numbers.Number):
            raise ValueError(f"a is not a number. Received a={a}")
        if not isinstance(weight_feature, str):
            raise ValueError(
                f"weight_feature must be a column name (str). Got {weight_feature}"
            )
        if comparison not in ("dataset", "complement"):
            raise ValueError(
                f"comparison was {comparison}, must be 'dataset' or 'complement'"
            )
        if direction not in ("more_compliant", "less_compliant", "both"):
            raise ValueError(
                f"direction was {direction}, must be 'more_compliant', "
                "'less_compliant', or 'both'"
            )
        if not 0.0 <= min_sg_coverage <= max_sg_coverage <= 1.0:
            raise ValueError(
                "require 0 <= min_sg_coverage <= max_sg_coverage <= 1; "
                f"got min={min_sg_coverage}, max={max_sg_coverage}"
            )
        self.a = a
        self.weight_feature = weight_feature
        self.comparison = comparison
        self.direction = direction
        self.min_sg_coverage = min_sg_coverage
        self.max_sg_coverage = max_sg_coverage
        self.invert = invert
        self.required_stat_attrs = ("size_sg", "weight_sg", "auc")
        self.all_target_values = None
        self.all_weight_values = None
        self.dataset_size = None
        self.has_constant_statistics = False

    def calculate_constant_statistics(self, data, target):
        """Cache the target vector, the weight-feature vector, and dataset size."""
        self.all_target_values = data[target.target_variable].to_numpy()
        self.all_weight_values = data[self.weight_feature].to_numpy()
        self.dataset_size = len(self.all_target_values)
        self.has_constant_statistics = True

    def _coverage_ok(self, sg_size):
        """True if sg_size falls within the allowed coverage band."""
        cov = sg_size / self.dataset_size
        return self.min_sg_coverage <= cov <= self.max_sg_coverage

    def _directed_deviation(self, auc):
        """Signed, direction-filtered deviation of AUC from 0.5 (>=0, or 0)."""
        if self.direction == "both":
            dev = abs(auc - 0.5)
        elif self.direction == "less_compliant":  # want AUC > 0.5
            dev = max(auc - 0.5, 0.0)
        else:  # more_compliant: want AUC < 0.5
            dev = max(0.5 - auc, 0.0)
        return 2.0 * dev

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        """Compute subgroup size, weight-sum, AUC, and a safe optimistic estimate.

        Out-of-coverage subgroups get auc=0.5 and estimate=-inf.
        """
        cover_arr, sg_size = ps.get_cover_array_and_size(
            subgroup, len(self.all_target_values), data
        )
        if (
            sg_size == 0
            or sg_size == self.dataset_size
            or not self._coverage_ok(sg_size)
        ):
            return self.tpl(sg_size, 0.0, 0.5, float("-inf"))
        sg_values = self.all_target_values[cover_arr]
        weight_sg = float(np.sum(self.all_weight_values[cover_arr]))
        if self.comparison == "complement":
            ref_values = self.all_target_values[~cover_arr]
        else:
            ref_values = self.all_target_values
        auc = _auc(sg_values, ref_values)
        # Safe (admissible, non-pruning) optimistic estimate: max deviation = 0.5,
        # max weight-sum is the total positive weight over the whole dataset.
        estimate = max(weight_sg, 0.0) ** self.a
        return self.tpl(sg_size, weight_sg, auc, estimate)

    def evaluate(self, subgroup, target, data, statistics=None):
        """Feature-sum-weighted, direction-filtered AUC deviation; 0 if filtered."""
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        if not self._coverage_ok(statistics.size_sg):
            return 0.0
        return statistics.weight_sg**self.a * self._directed_deviation(statistics.auc)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        """Return the cached optimistic estimate (loose but admissible)."""
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate

class EMDQFNumeric(ps.BoundedInterestingnessMeasure):
    """Earth Mover's Distance quality function for an ordinal target.

    For a 1-D target, EMD is the L1 distance between the empirical CDFs, using
    the true numeric gaps between consecutive target values as ground distance.
    EMD itself is directionless; direction is inferred from the sign of
    (subgroup mean - reference mean) with severity coded 0 (best) .. K (worst):
    mean_sg < mean_ref => MORE compliant, mean_sg > mean_ref => LESS compliant.

        phi(sg) = n_sg**a * EMD(sg, ref)

    Subgroups going against the requested direction score 0. Subgroups outside
    the coverage band [min_sg_coverage, max_sg_coverage] are discarded.

    Parameters:
        a (float): exponent weighting subgroup size.
        comparison (str): 'dataset' (standard) or 'complement'.
        direction (str): 'more_compliant', 'less_compliant', or 'both'.
        min_sg_coverage (float): minimum dataset fraction a subgroup must cover.
        max_sg_coverage (float): maximum dataset fraction a subgroup may cover.
        invert (bool): kept for interface parity, unused.
    """

    tpl = namedtuple(
        "EMDQFNumeric_parameters", ("size_sg", "emd", "sign", "estimate")
    )

    def __init__(
        self,
        a,
        comparison="dataset",
        direction="both",
        min_sg_coverage=0.0,
        max_sg_coverage=1.0,
        invert=False,
    ):
        if not isinstance(a, numbers.Number):
            raise ValueError(f"a is not a number. Received a={a}")
        if comparison not in ("dataset", "complement"):
            raise ValueError(
                f"comparison was {comparison}, must be 'dataset' or 'complement'"
            )
        if direction not in ("more_compliant", "less_compliant", "both"):
            raise ValueError(
                f"direction was {direction}, must be 'more_compliant', "
                "'less_compliant', or 'both'"
            )
        if not 0.0 <= min_sg_coverage <= max_sg_coverage <= 1.0:
            raise ValueError(
                "require 0 <= min_sg_coverage <= max_sg_coverage <= 1; "
                f"got min={min_sg_coverage}, max={max_sg_coverage}"
            )
        self.a = a
        self.comparison = comparison
        self.direction = direction
        self.min_sg_coverage = min_sg_coverage
        self.max_sg_coverage = max_sg_coverage
        self.invert = invert
        self.required_stat_attrs = ("size_sg", "emd", "sign")
        self.all_target_values = None
        self.grid = None
        self.dataset_size = None
        self.dataset_mean = None
        self.has_constant_statistics = False

    def calculate_constant_statistics(self, data, target):
        """Cache the target vector, dataset size, value grid, and dataset mean."""
        self.all_target_values = data[target.target_variable].to_numpy()
        self.dataset_size = len(self.all_target_values)
        self.grid = np.unique(self.all_target_values)
        self.dataset_mean = float(np.mean(self.all_target_values))
        self.has_constant_statistics = True

    def _coverage_ok(self, sg_size):
        """True if sg_size falls within the allowed coverage band."""
        cov = sg_size / self.dataset_size
        return self.min_sg_coverage <= cov <= self.max_sg_coverage

    def _direction_ok(self, sign):
        """True if the subgroup's shift sign matches the requested direction.

        sign = +1 means less compliant (higher mean severity), -1 more compliant.
        """
        if self.direction == "both":
            return True
        if self.direction == "less_compliant":
            return sign > 0
        return sign < 0  # more_compliant

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        """Compute subgroup size, EMD, shift sign, and a safe optimistic estimate.

        Out-of-coverage subgroups get emd=0, sign=0 and estimate=-inf.
        """
        cover_arr, sg_size = ps.get_cover_array_and_size(
            subgroup, len(self.all_target_values), data
        )
        if (
            sg_size == 0
            or sg_size == self.dataset_size
            or not self._coverage_ok(sg_size)
        ):
            return self.tpl(sg_size, 0.0, 0, float("-inf"))
        sg_values = self.all_target_values[cover_arr]
        if self.comparison == "dataset":
            ref_values = self.all_target_values
            ref_mean = self.dataset_mean
        else:
            ref_values = self.all_target_values[~cover_arr]
            ref_mean = float(np.mean(ref_values))
        emd = _emd(sg_values, ref_values, self.grid)
        sg_mean = float(np.mean(sg_values))
        sign = int(np.sign(sg_mean - ref_mean))  # +1 less compliant, -1 more
        # Safe (admissible, non-pruning) optimistic estimate: max EMD = grid span.
        # (Note: direction is not exploited in the bound; see caveat below.)
        max_emd = float(self.grid[-1] - self.grid[0])
        estimate = sg_size**self.a * max_emd
        return self.tpl(sg_size, emd, sign, estimate)

    def evaluate(self, subgroup, target, data, statistics=None):
        """Size-weighted EMD; 0 if out of coverage band or wrong direction."""
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        if not self._coverage_ok(statistics.size_sg):
            return 0.0
        if not self._direction_ok(statistics.sign):
            return 0.0
        return statistics.size_sg**self.a * statistics.emd

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        """Return the cached optimistic estimate (loose but admissible)."""
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate