import pandas as pd
import numpy as np
import collections

pos_columns = ["x", "y"]
t_column = "frame"
trackno_column = "particle"


def calculate_sd(data, tlag, pixel_size, tlag_thresh = (0, np.inf),
                 matlab_compat = True,
                 pos_columns=pos_columns,
                 t_column=t_column,
                 trackno_column=trackno_column):
    sd_dict = collections.OrderedDict()
    for pn in data[trackno_column].unique():
        pdata = data.loc[data[trackno_column] == pn] #data for current track

        #fill gaps with NaNs
        frame_nos = pdata[t_column].astype(int)
        start_frame = min(frame_nos)
        end_frame = max(frame_nos)
        frame_list = pd.DataFrame(list(range(start_frame, end_frame + 1)),
                                  columns=[t_column])

        pdata = pd.merge(pdata, frame_list, on=t_column, how="outer",
                         sort=True)

        #the original msdplot matlab tool throws away all long trajectories
        if (matlab_compat
            and (not tlag_thresh[0] <= len(pdata) <= tlag_thresh[1] + 1)):
            continue

        pdata = pdata[pos_columns].as_matrix()

        for i in range(round(max(1, tlag_thresh[0])),
                       round(min(len(pdata), tlag_thresh[1])) + 1):
            #prepend/append i NaNs to coordinate list and calculate differences
            padding = [[np.NaN]*len(pos_columns)]*i
            disp = np.vstack((pdata, padding)) - np.vstack((padding, pdata))
            #calculate sds
            sds = np.sum(disp**2, axis=1)
            #get rid of NaNs
            sds = sds[~np.isnan(sds)]
            #append to output structure
            sd_dict[i*tlag] = np.concatenate((sd_dict.get(i*tlag, []),
                                              sds * pixel_size**2))

    return sd_dict


def calculate_msd(data, tlag, pixel_size, tlag_thresh = (0, np.inf),
                 matlab_compat = True,
                 pos_columns=pos_columns,
                 t_column=t_column,
                 trackno_column=trackno_column):
    sds = calculate_sd(data, tlag, pixel_size, tlag_thresh, matlab_compat,
                       pos_columns, t_column, trackno_column)

    ret = collections.OrderedDict()
    ret["tlag"] = list(sds.keys())
    sval = sds.values()
    ret["msd"] = [sd.mean() for sd in sval]
    ret["stderr"] = [sd.std(ddof=1)/np.sqrt(len(sd)) for sd in sval]
    #TODO: Quian errors
    ret = pd.DataFrame(ret)
    ret.sort("tlag", inplace=True)
    ret.reset_index(inplace=True, drop=True)
    return ret