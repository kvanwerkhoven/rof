'''
ROF evaluation configuration file

Code by Katie van Werkhoven, ISED
'''

from datetime import datetime, timedelta
import importlib
from pathlib import Path

# import functions created for this evaluation
# force reload of nwm to ensure latest version (needed during testing when still making edits to the functions)
import nwm_utilities as nwm
importlib.reload(nwm)


###############################################################################################
#  Global settings - used throughout all steps
###############################################################################################

# geographic domain as defined by separate sets of NWM output files (domain must be same as the tag in filenames prior to ".nc"
domain = 'conus'
#domain = 'hawaii'

# Note nwm version determined based on reference datetime (v2.0 prior to 4/20 13z, v2.1 after)
# currently does not work for a range that spans the version change

# local nwm netcdf storage (grids and channel), currently an external drive
nwm_repo = Path('E:/NWM_Cache')
    
# input/output directories
in_dir = Path('C:/repos/git/nwm/input')
out_dir = Path('C:/repos/git/nwm/output')

###############################################################################################
#  Evaluation specs - configuration used, type of eval, timing of eval
###############################################################################################

# NWM configuration being evaluated (full name used in to NWM filenames) 
# (e.g. "short_range", "medium_range", "analysis_assim", "analysis_assim_extend")
# (currently for ROF, 'short_range' is only tested option)

eval_config = 'short_range'       

# Source of 'verifying observations', i.e., comparison data, options are:
# "analysis_assim" - standard analysis and assimilation only
# "analysis_assim_extend" - extended analysis and assimilation only (not available for Hawaii)
# "latest_ana" - best available AnA for each timestep (extended if avail, otherwise standard)

verif_config = 'latest_ana'
#verif_config = 'analysis_assim_extend'

# Timing of evaluation - dictates which data/reftimes are available to use
# "current" - run current (most recent) reference time possible (depends on verif-config)
# "past"  - any date or range of dates in the past

eval_timing = "current"
#eval_timing = "past"

# date range for timing method "past", for other timing methods these dates are ignored

start_time = datetime(2021, 5, 6, 5, 0, 0)
end_time = datetime(2021, 5, 6, 5, 0, 0)

# Evaluation metric
metric = 'rof'

# Spatial aggregation method - ("str_length") stream length or number of reaches ("str_num")
spatial_agg_method = "str_length"

# Stream order limit, if any - upper limit on order of streams to include in evaluation (0 if none)
order_max = 4 

# event threshold for defined metric and spatial agg method (e.g. 30% of reaches in ROF)
event_thresh = 30

# method of MAP calculation in zonal_stats: all_touched = True or False
all_touched = False

# use existing MAP values previously calculated if they exist
use_existing = True

# Execute evaluation
nwm.rof_eval_with_download(domain, nwm_repo, in_dir, out_dir, 
                           eval_config, verif_config, eval_timing,
                           metric, spatial_agg_method, event_thresh, order_max,
                           use_existing = use_existing, all_touched = all_touched,
                           start = start_time, end = end_time)