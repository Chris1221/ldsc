"""
This is an updated version of S-LDSC that can be run in a python environment.
"""
import ldscore.ldscore as ld
import ldscore.parse as ps
import ldscore.sumstats as sumstats
import ldscore.regressions as reg
import numpy as np
import pandas as pd
from subprocess import call
from itertools import product
import time, sys, traceback, argparse
import argparse
import sys
import time
import traceback
from functools import reduce
from typing import Optional


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


try:
    x = pd.DataFrame({"A": [1, 2, 3]})
    x.sort_values(by="A")
except AttributeError:
    raise ImportError("LDSC requires pandas version >= 0.17.0")

__version__ = "1.0.1"
MASTHEAD = "*********************************************************************\n"
MASTHEAD += "* LD Score Regression (LDSC)\n"
MASTHEAD += "* Version {V}\n".format(V=__version__)
MASTHEAD += "* (C) 2014-2019 Brendan Bulik-Sullivan and Hilary Finucane\n"
MASTHEAD += "* Broad Institute of MIT and Harvard / MIT Department of Mathematics\n"
MASTHEAD += "* GNU General Public License v3\n"
MASTHEAD += "*********************************************************************\n"
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("precision", 4)
pd.set_option("max_colwidth", 1000)
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=4)


def sec_to_str(t):
    """Convert seconds to days:hours:minutes:seconds"""
    [d, h, m, s, n] = reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24]
    )
    f = ""
    if d > 0:
        f += "{D}d:".format(D=d)
    if h > 0:
        f += "{H}h:".format(H=h)
    if m > 0:
        f += "{M}m:".format(M=m)

    f += "{S}s".format(S=s)
    return f


def _remove_dtype(x):
    """Removes dtype: float64 and dtype: int64 from pandas printouts"""
    x = str(x)
    x = x.replace("\ndtype: int64", "")
    x = x.replace("\ndtype: float64", "")
    return x


class Logger(object):
    """
    Lightweight logging.
    TODO: replace with logging module

    """

    def __init__(self, fh):
        self.log_fh = open(fh, "wb")

    def log(self, msg):
        """
        Print to log file and stdout with a single command.

        """
        print(msg, file=self.log_fh)
        print(msg)


def __filter__(fname, noun, verb, merge_obj):
    merged_list = None
    if fname:
        f = lambda x, n: x.format(noun=noun, verb=verb, fname=fname, num=n)
        x = ps.FilterFile(fname)
        c = "Read list of {num} {noun} to {verb} from {fname}"
        print(f(c, len(x.IDList)))
        merged_list = merge_obj.loj(x.IDList)
        len_merged_list = len(merged_list)
        if len_merged_list > 0:
            c = "After merging, {num} {noun} remain"
            print(f(c, len_merged_list))
        else:
            error_msg = "No {noun} retained for analysis"
            raise ValueError(f(error_msg, 0))

        return merged_list


def annot_sort_key(s):
    """For use with --cts-bin. Fixes weird pandas crosstab column order."""
    if type(s) == tuple:
        s = [x.split("_")[0] for x in s]
        s = map(lambda x: float(x) if x != "min" else -float("inf"), s)
    else:  # type(s) = str:
        s = s.split("_")[0]
        if s == "min":
            s = float("-inf")
        else:
            s = float(s)

    return s


def ldscore(args, log):
    """
    Wrapper function for estimating l1, l1^2, l2 and l4 (+ optionally standard errors) from
    reference panel genotypes.

    Annot format is
    chr snp bp cm <annotations>

    """

    if args.bfile:
        snp_file, snp_obj = args.bfile + ".bim", ps.PlinkBIMFile
        ind_file, ind_obj = args.bfile + ".fam", ps.PlinkFAMFile
        array_file, array_obj = args.bfile + ".bed", ld.PlinkBEDFile

    # read bim/snp
    array_snps = snp_obj(snp_file)
    m = len(array_snps.IDList)
    log.log("Read list of {m} SNPs from {f}".format(m=m, f=snp_file))
    if args.annot is not None:  # read --annot
        try:
            if args.thin_annot:  # annot file has only annotations
                annot = ps.ThinAnnotFile(args.annot)
                n_annot, ma = len(annot.df.columns), len(annot.df)
                log.log(
                    "Read {A} annotations for {M} SNPs from {f}".format(
                        f=args.annot, A=n_annot, M=ma
                    )
                )
                annot_matrix = annot.df.values
                annot_colnames = annot.df.columns
                keep_snps = None
            else:
                annot = ps.AnnotFile(args.annot)
                n_annot, ma = len(annot.df.columns) - 4, len(annot.df)
                log.log(
                    "Read {A} annotations for {M} SNPs from {f}".format(
                        f=args.annot, A=n_annot, M=ma
                    )
                )
                annot_matrix = np.array(annot.df.iloc[:, 4:])
                annot_colnames = annot.df.columns[4:]
                keep_snps = None
                if np.any(annot.df.SNP.values != array_snps.df.SNP.values):
                    raise ValueError(
                        "The .annot file must contain the same SNPs in the same"
                        + " order as the .bim file."
                    )
        except Exception:
            log.log("Error parsing .annot file")
            raise

    elif args.extract is not None:  # --extract
        keep_snps = __filter__(args.extract, "SNPs", "include", array_snps)
        annot_matrix, annot_colnames, n_annot = None, None, 1

    elif args.cts_bin is not None and args.cts_breaks is not None:  # --cts-bin
        cts_fnames = sumstats._splitp(args.cts_bin)  # read filenames
        args.cts_breaks = args.cts_breaks.replace(
            "N", "-"
        )  # replace N with negative sign
        try:  # split on x
            breaks = [
                [float(x) for x in y.split(",")] for y in args.cts_breaks.split("x")
            ]
        except ValueError as e:
            raise ValueError(
                "--cts-breaks must be a comma-separated list of numbers: " + str(e.args)
            )

        if len(breaks) != len(cts_fnames):
            raise ValueError(
                "Need to specify one set of breaks for each file in --cts-bin."
            )

        if args.cts_names:
            cts_colnames = [str(x) for x in args.cts_names.split(",")]
            if len(cts_colnames) != len(cts_fnames):
                msg = "Must specify either no --cts-names or one value for each file in --cts-bin."
                raise ValueError(msg)

        else:
            cts_colnames = ["ANNOT" + str(i) for i in range(len(cts_fnames))]

        log.log(
            "Reading numbers with which to bin SNPs from {F}".format(F=args.cts_bin)
        )

        cts_levs = []
        full_labs = []
        for i, fh in enumerate(cts_fnames):
            vec = ps.read_cts(cts_fnames[i], array_snps.df.SNP.values)

            max_cts = np.max(vec)
            min_cts = np.min(vec)
            cut_breaks = list(breaks[i])
            name_breaks = list(cut_breaks)
            if np.all(cut_breaks >= max_cts) or np.all(cut_breaks <= min_cts):
                raise ValueError(
                    "All breaks lie outside the range of the cts variable."
                )

            if np.all(cut_breaks <= max_cts):
                name_breaks.append(max_cts)
                cut_breaks.append(max_cts + 1)

            if np.all(cut_breaks >= min_cts):
                name_breaks.append(min_cts)
                cut_breaks.append(min_cts - 1)

            name_breaks.sort()
            cut_breaks.sort()
            n_breaks = len(cut_breaks)
            # so that col names are consistent across chromosomes with different max vals
            name_breaks[0] = "min"
            name_breaks[-1] = "max"
            name_breaks = [str(x) for x in name_breaks]
            labs = [
                name_breaks[i] + "_" + name_breaks[i + 1] for i in range(n_breaks - 1)
            ]
            cut_vec = pd.Series(pd.cut(vec, bins=cut_breaks, labels=labs))
            cts_levs.append(cut_vec)
            full_labs.append(labs)

        annot_matrix = pd.concat(cts_levs, axis=1)
        annot_matrix.columns = cts_colnames
        # crosstab -- for now we keep empty columns
        annot_matrix = pd.crosstab(
            annot_matrix.index,
            [annot_matrix[i] for i in annot_matrix.columns],
            dropna=False,
            colnames=annot_matrix.columns,
        )

        # add missing columns
        if len(cts_colnames) > 1:
            for x in product(*full_labs):
                if x not in annot_matrix.columns:
                    annot_matrix[x] = 0
        else:
            for x in full_labs[0]:
                if x not in annot_matrix.columns:
                    annot_matrix[x] = 0

        annot_matrix = annot_matrix[sorted(annot_matrix.columns, key=annot_sort_key)]
        if len(cts_colnames) > 1:
            # flatten multi-index
            annot_colnames = [
                "_".join([cts_colnames[i] + "_" + b for i, b in enumerate(c)])
                for c in annot_matrix.columns
            ]
        else:
            annot_colnames = [cts_colnames[0] + "_" + b for b in annot_matrix.columns]

        annot_matrix = np.matrix(annot_matrix)
        keep_snps = None
        n_annot = len(annot_colnames)
        if np.any(np.sum(annot_matrix, axis=1) == 0):
            # This exception should never be raised. For debugging only.
            raise ValueError(
                "Some SNPs have no annotation in --cts-bin. This is a bug!"
            )

    else:
        annot_matrix, annot_colnames, keep_snps = (
            None,
            None,
            None,
        )
        n_annot = 1

    # read fam
    array_indivs = ind_obj(ind_file)
    n = len(array_indivs.IDList)
    log.log("Read list of {n} individuals from {f}".format(n=n, f=ind_file))
    # read keep_indivs
    if args.keep:
        keep_indivs = __filter__(args.keep, "individuals", "include", array_indivs)
    else:
        keep_indivs = None

    # read genotype array
    log.log("Reading genotypes from {fname}".format(fname=array_file))
    geno_array = array_obj(
        array_file,
        n,
        array_snps,
        keep_snps=keep_snps,
        keep_indivs=keep_indivs,
        mafMin=args.maf,
    )

    # filter annot_matrix down to only SNPs passing MAF cutoffs
    if annot_matrix is not None:
        annot_keep = geno_array.kept_snps
        annot_matrix = annot_matrix[annot_keep, :]

    # determine block widths
    x = np.array((args.ld_wind_snps, args.ld_wind_kb, args.ld_wind_cm), dtype=bool)
    if np.sum(x) != 1:
        raise ValueError("Must specify exactly one --ld-wind option")

    if args.ld_wind_snps:
        max_dist = args.ld_wind_snps
        coords = np.array(range(geno_array.m))
    elif args.ld_wind_kb:
        max_dist = args.ld_wind_kb * 1000
        coords = np.array(array_snps.df["BP"])[geno_array.kept_snps]
    elif args.ld_wind_cm:
        max_dist = args.ld_wind_cm
        coords = np.array(array_snps.df["CM"])[geno_array.kept_snps]

    block_left = ld.getBlockLefts(coords, max_dist)
    if block_left[len(block_left) - 1] == 0 and not args.yes_really:
        error_msg = (
            "Do you really want to compute whole-chomosome LD Score? If so, set the "
        )
        error_msg += "--yes-really flag (warning: it will use a lot of time / memory)"
        raise ValueError(error_msg)

    scale_suffix = ""
    if args.pq_exp is not None:
        log.log("Computing LD with pq ^ {S}.".format(S=args.pq_exp))
        msg = "Note that LD Scores with pq raised to a nonzero power are"
        msg += "not directly comparable to normal LD Scores."
        log.log(msg)
        scale_suffix = "_S{S}".format(S=args.pq_exp)
        pq = np.matrix(geno_array.maf * (1 - geno_array.maf)).reshape((geno_array.m, 1))
        pq = np.power(pq, args.pq_exp)

        if annot_matrix is not None:
            annot_matrix = np.multiply(annot_matrix, pq)
        else:
            annot_matrix = pq

    log.log("Estimating LD Score.")
    lN = geno_array.ldScoreVarBlocks(block_left, args.chunk_size, annot=annot_matrix)
    col_prefix = "L2"
    file_suffix = "l2"

    if n_annot == 1:
        ldscore_colnames = [col_prefix + scale_suffix]
    else:
        ldscore_colnames = [y + col_prefix + scale_suffix for y in annot_colnames]

    # print .ldscore. Output columns: CHR, BP, RS, [LD Scores]
    out_fname = args.out + "." + file_suffix + ".ldscore"
    new_colnames = geno_array.colnames + ldscore_colnames
    df = pd.DataFrame.from_records(np.c_[geno_array.df, lN])
    df.columns = new_colnames
    if args.print_snps:
        if args.print_snps.endswith("gz"):
            print_snps = pd.read_csv(args.print_snps, header=None, compression="gzip")
        elif args.print_snps.endswith("bz2"):
            print_snps = pd.read_csv(args.print_snps, header=None, compression="bz2")
        else:
            print_snps = pd.read_csv(args.print_snps, header=None)
        if len(print_snps.columns) > 1:
            raise ValueError(
                "--print-snps must refer to a file with a one column of SNP IDs."
            )
        log.log(
            "Reading list of {N} SNPs for which to print LD Scores from {F}".format(
                F=args.print_snps, N=len(print_snps)
            )
        )

        print_snps.columns = ["SNP"]
        df = df.ix[df.SNP.isin(print_snps.SNP), :]
        if len(df) == 0:
            raise ValueError("After merging with --print-snps, no SNPs remain.")
        else:
            msg = "After merging with --print-snps, LD Scores for {N} SNPs will be printed."
            log.log(msg.format(N=len(df)))

    l2_suffix = ".gz"
    log.log("Writing LD Scores for {N} SNPs to {f}.gz".format(f=out_fname, N=len(df)))
    df.drop(["CM", "MAF"], axis=1).to_csv(
        out_fname, sep="\t", header=True, index=False, float_format="%.3f"
    )
    call(["gzip", "-f", out_fname])
    if annot_matrix is not None:
        M = np.atleast_1d(np.squeeze(np.asarray(np.sum(annot_matrix, axis=0))))
        ii = geno_array.maf > 0.05
        M_5_50 = np.atleast_1d(
            np.squeeze(np.asarray(np.sum(annot_matrix[ii, :], axis=0)))
        )
    else:
        M = [geno_array.m]
        M_5_50 = [np.sum(geno_array.maf > 0.05)]

    # print .M
    fout_M = open(args.out + "." + file_suffix + ".M", "wb")
    print >> fout_M, "\t".join(map(str, M))
    fout_M.close()

    # print .M_5_50
    fout_M_5_50 = open(args.out + "." + file_suffix + ".M_5_50", "wb")
    print >> fout_M_5_50, "\t".join(map(str, M_5_50))
    fout_M_5_50.close()

    # print annot matrix
    if (args.cts_bin is not None) and not args.no_print_annot:
        out_fname_annot = args.out + ".annot"
        new_colnames = geno_array.colnames + ldscore_colnames
        annot_df = pd.DataFrame(np.c_[geno_array.df, annot_matrix])
        annot_df.columns = new_colnames
        del annot_df["MAF"]
        log.log(
            "Writing annot matrix produced by --cts-bin to {F}".format(
                F=out_fname + ".gz"
            )
        )
        annot_df.to_csv(out_fname_annot, sep="\t", header=True, index=False)
        call(["gzip", "-f", out_fname_annot])

    # print LD Score summary
    pd.set_option("display.max_rows", 200)
    log.log("\nSummary of LD Scores in {F}".format(F=out_fname + l2_suffix))
    t = df.ix[:, 4:].describe()
    log.log(t.ix[1:, :])

    np.seterr(divide="ignore", invalid="ignore")  # print NaN instead of weird errors
    # print correlation matrix including all LD Scores and sample MAF
    log.log("")
    log.log("MAF/LD Score Correlation Matrix")
    log.log(df.ix[:, 4:].corr())

    # print condition number
    if (
        n_annot > 1
    ):  # condition number of a column vector w/ nonzero var is trivially one
        log.log("\nLD Score Matrix Condition Number")
        cond_num = np.linalg.cond(df.ix[:, 5:])
        log.log(reg.remove_brackets(str(np.matrix(cond_num))))
        if cond_num > 10000:
            log.log("WARNING: ill-conditioned LD Score Matrix!")

    # summarize annot matrix if there is one
    if annot_matrix is not None:
        # covariance matrix
        x = pd.DataFrame(annot_matrix, columns=annot_colnames)
        log.log("\nAnnotation Correlation Matrix")
        log.log(x.corr())

        # column sums
        log.log("\nAnnotation Matrix Column Sums")
        log.log(_remove_dtype(x.sum(axis=0)))

        # row sums
        log.log("\nSummary of Annotation Matrix Row Sums")
        row_sums = x.sum(axis=1).describe()
        log.log(_remove_dtype(row_sums))

    np.seterr(divide="raise", invalid="raise")


def ldsc_analysis(
    out: str,
    bfile: Optional[str] = None,
    l2: Optional[str] = None,
    annot: Optional[str] = None,
    extract: Optional[str] = None,
    cts_bin: Optional[str] = None,
    cts_breaks: Optional[str] = None,
    per_allele: Optional[bool] = False,
    pq_exp: Optional[float] = None,
    h2: Optional[bool] = False,
    h2_cts: Optional[str] = None,
    rg: Optional[str] = None,
    ref_ld: Optional[str] = None,
    ref_ld_chr: Optional[str] = None,
    w_ld: Optional[str] = None,
    w_ld_chr: Optional[str] = None,
    overlap_annot: Optional[bool] = False,
    print_coefficients: Optional[bool] = False,
    frqfile: Optional[str] = None,
    frqfile_chr: Optional[str] = None,
    no_intercept: Optional[bool] = False,
    intercept_h2: Optional[str] = None,
    intercept_gencov: Optional[str] = None,
    M: Optional[str] = None,
    two_step: Optional[float] = None,
    chisq_max: Optional[float] = None,
    ref_ld_chr_cts: Optional[str] = None,
    print_all_cts: Optional[bool] = False,
    print_cov: Optional[bool] = False,
    print_delete_vals: Optional[bool] = False,
    chunk_size: Optional[int] = 50,
    pickle: Optional[bool] = False,
    yes_really: Optional[bool] = False,
    invert_anyway: Optional[bool] = False,
    n_blocks: Optional[int] = 200,
    not_M_5_50: Optional[bool] = False,
    return_silly_things: Optional[bool] = False,
    no_check_alleles: Optional[bool] = False,
    samp_prev: Optional[str] = None,
    pop_prev: Optional[str] = None,
) -> None:
    log = Logger(out + ".log")

    # This is a very silly workaround because I do not have time to
    # recode the arg access in the various submodules (yet).
    args = DotDict(locals())

    log.log("Beginning analysis at {T}".format(T=time.ctime()))

    start_time = time.time()
    if n_blocks <= 1:
        raise ValueError("--n-blocks must be an integer > 1.")
    if bfile is not None:
        if l2 is None:
            raise ValueError("Must specify --l2 with --bfile.")
        if annot is not None and extract is not None:
            raise ValueError("--annot and --extract are currently incompatible.")
        if cts_bin is not None and extract is not None:
            raise ValueError("--cts-bin and --extract are currently incompatible.")
        if annot is not None and cts_bin is not None:
            raise ValueError("--annot and --cts-bin are currently incompatible.")
        if (cts_bin is not None) != (cts_breaks is not None):
            raise ValueError("Must set both or neither of --cts-bin and --cts-breaks.")
        if per_allele and pq_exp is not None:
            raise ValueError(
                "Cannot set both --per-allele and --pq-exp (--per-allele is equivalent to --pq-exp 1)."
            )
        if per_allele:
            pq_exp = 1

        ldscore(args, log)

    # summary statistics
    elif (h2 or rg or h2_cts) and (ref_ld or ref_ld_chr) and (w_ld or w_ld_chr):
        if h2 is not None and rg is not None:
            raise ValueError("Cannot set both --h2 and --rg.")
        if ref_ld and ref_ld_chr:
            raise ValueError("Cannot set both --ref-ld and --ref-ld-chr.")
        if w_ld and w_ld_chr:
            raise ValueError("Cannot set both --w-ld and --w-ld-chr.")
        if (samp_prev is not None) != (pop_prev is not None):
            raise ValueError("Must set both or neither of --samp-prev and --pop-prev.")

        if not overlap_annot or not_M_5_50:
            if frqfile is not None or frqfile_chr is not None:
                log.log("The frequency file is unnecessary and is being ignored.")
                frqfile = None
                frqfile_chr = None
        if overlap_annot and not not_M_5_50:
            if not ((frqfile and ref_ld) or (frqfile_chr and ref_ld_chr)):
                raise ValueError(
                    "Must set either --frqfile and --ref-ld or --frqfile-chr and --ref-ld-chr"
                )

        if rg:
            sumstats.estimate_rg(args, log)
        elif h2:
            sumstats.estimate_h2(args, log)
        elif h2_cts:
            sumstats.cell_type_specific(args, log)
