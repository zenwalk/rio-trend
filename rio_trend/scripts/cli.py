"""Main CLI."""
import sys

import click
import numpy as np
import rasterio
from rasterio.rio.options import creation_options
from rasterio.transform import guard_transform
from rio_trend.workers import color_worker
from rio_trend.operations import parse_operations, simple_atmo_opstring
import riomucho

jobs_opt = click.option(
    "--jobs",
    "-j",
    type=int,
    default=1,
    help="Number of jobs to run simultaneously, Use -1 for all cores, default: 1",
)


def check_jobs(jobs):
    """Validate number of jobs."""
    if jobs == 0:
        raise click.UsageError("Jobs must be >= 1 or == -1")
    elif jobs < 0:
        import multiprocessing
        jobs = multiprocessing.cpu_count()
    return jobs


@click.command("trend")
@jobs_opt
@click.option(
    "--out-dtype",
    "-d",
    type=click.Choice(["uint8", "uint16"]),
    help="Integer data type for output data, default: same as input",
)
# @click.option(
#     '--mask-path',
#     "-m",
#     type=click.Path(exists=True),
#     help="mask file"
# )
@click.argument("src_path", type=click.Path(exists=True))
@click.argument("dst_path", type=click.Path(exists=False))
@click.pass_context
@creation_options
def trend(ctx, jobs, out_dtype, src_path, dst_path, creation_options):
    """long help
    """

    with rasterio.open(src_path) as src:
        opts = src.profile.copy()
        windows = [(window, ij) for ij, window in src.block_windows()]

    opts.update(**creation_options)
    opts["transform"] = guard_transform(opts["transform"])

    out_dtype = out_dtype if out_dtype else opts["dtype"]
    opts["dtype"] = rasterio.float32

    nodata = np.finfo(np.float32).min
    opts.update(count=3, nodata=nodata)

    args = {"out_dtype": out_dtype, "nodatavals": src.nodatavals, "nodata": src.nodata}

    # Just run this for validation this time
    # parsing will be run again within the worker
    # where its returned value will be used
    try:
        # parse_operations(args["ops_string"])
        pass
    except ValueError as e:
        raise click.UsageError(str(e))

    jobs = check_jobs(jobs)

    print(jobs)

    if jobs > 1:
        with riomucho.RioMucho(
            [src_path],
            dst_path,
            color_worker,
            windows=windows,
            options=opts,
            global_args=args,
            mode="manual_read",
        ) as mucho:
            mucho.run(jobs)
    else:
        with rasterio.open(dst_path, "w", **opts) as dest:
            with rasterio.open(src_path) as src:
                rasters = [src]
                for window, ij in windows:
                    arr = color_worker(rasters, window, ij, args)
                    dest.write(arr, window=window)

                # dest.colorinterp = src.colorinterp
